import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
import time
import horovod.tensorflow as hvd
import utils
import tensorflow_probability as tfp
from deepsets_cond import DeepSetsAtt, Resnet
from tensorflow.keras.activations import swish, relu
import gc
from tensorflow_probability.python.math.diag_jacobian import diag_jacobian
#tf.random.set_seed(1234)

class GSGM(keras.Model):
    """Score based generative model"""
    def __init__(self,name='SGM',npart=100,
                 particle='gluon_tagging',
                 ll_training=False,config=None):
        super(GSGM, self).__init__()

        self.config = config
        if config is None:
            raise ValueError("Config file not given")

        self.activation = layers.LeakyReLU(alpha=0.01)
        
        
        #self.activation = layers.LeakyReLU(alpha=0.01)
        self.num_feat = self.config['NUM_FEAT']
        self.num_jet = self.config['NUM_JET']
        
        self.num_embed = self.config['EMBED']
        self.max_part = npart
        self.particle = particle
        self.ll_training = ll_training
        self.ema=0.999
        self.minlogsnr = -10.0
        self.maxlogsnr = 10.0

        
        self.projection = self.GaussianFourierProjection(scale = 16)
        self.loss_tracker = keras.metrics.Mean(name="loss")


        #Transformation applied to conditional inputs
        inputs_time = Input((1))
        inputs_jet = Input((self.num_jet))
        inputs_particles = Input((None,2*self.num_feat))
        inputs_mask = Input((None,1)) #mask to identify zero-padded objects


        graph_conditional = self.Embedding(inputs_time,self.projection)
        jet_conditional = self.Embedding(inputs_time,self.projection)
        
        dense_jet = layers.Dense(self.num_embed,activation=None)(inputs_jet) 
        dense_jet = self.activation(dense_jet)     

        graph_conditional = layers.Dense(self.num_embed,activation=None)(tf.concat(
            [graph_conditional,dense_jet],-1))
        graph_conditional=self.activation(graph_conditional)
        
        jet_conditional = layers.Dense(self.num_embed,activation=None)(jet_conditional)
        jet_conditional=self.activation(jet_conditional)

        
        self.shape = (-1,1,1)
        inputs,outputs = DeepSetsAtt(
            inputs_particles,
            num_feat=self.num_feat,
            time_embedding=graph_conditional,
            num_heads=2,
            num_transformer = 6,
            projection_dim = 128,
            mask = inputs_mask,
        )

        self.model_part = keras.Model(inputs=[inputs,inputs_time,inputs_jet,inputs_mask],
                                      outputs=outputs)
        
        outputs = Resnet(
            inputs_jet,
            self.num_jet,
            jet_conditional,
            num_embed=self.num_embed,
            num_layer = 3,
            mlp_dim= 256,
        )

        self.model_jet = keras.Model(inputs=[inputs_jet,inputs_time],
                                     outputs=outputs)


        self.ema_jet = keras.models.clone_model(self.model_jet)
        self.ema_part = keras.models.clone_model(self.model_part)
        
        
    @property
    def metrics(self):
        """List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker]
    

    def GaussianFourierProjection(self,scale = 30):
        #half_dim = 16
        half_dim = self.num_embed // 2
        emb = tf.math.log(10000.0) / (half_dim - 1)
        emb = tf.cast(emb,tf.float32)
        freq = tf.exp(-emb* tf.range(start=0, limit=half_dim, dtype=tf.float32))
        return freq


    def Embedding(self,inputs,projection):
        angle = inputs*projection*1000
        embedding = tf.concat([tf.math.sin(angle),tf.math.cos(angle)],-1)
        embedding = layers.Dense(2*self.num_embed,activation=None)(embedding)
        embedding = self.activation(embedding)
        embedding = layers.Dense(self.num_embed)(embedding)
        return embedding

    def prior_sde(self,dimensions):
        return tf.random.normal(dimensions,dtype=tf.float32)


    def Featurizer(self,features):
        z = 1.0 - tf.exp(features[:,:,2])
        dr = tf.sqrt(features[:,:,0]**2 + features[:,:,1]**2)
        e2 =  (1.0 + tf.math.sinh(features[:,:,0])**2)*z**2
        add_features = tf.stack([z,dr,e2],-1)
        return tf.concat([features,add_features],-1)
    
    def FF(self,features,expand=False):
        #Gaussian features to the inputs
        max_proj = 8
        min_proj = 6
        freq = tf.range(start=min_proj, limit=max_proj, dtype=tf.float32)
        freq = 2.**(freq) * 2 * np.pi        
        x = features
        freq = tf.tile(freq[None, :], ( 1, tf.shape(x)[-1]))  
        h = tf.repeat(x, max_proj-min_proj, axis=-1)
        if expand:
            angle = h*freq[None,:]
        else:
            angle = h*freq
        h = tf.concat([tf.math.sin(angle),tf.math.cos(angle)],-1)
        return tf.concat([features,h],-1)

    @tf.function
    def logsnr_schedule_cosine(self,t, logsnr_min, logsnr_max):
        #if self.ll_training:return logsnr_max - tf.cast(t,tf.float32)*(logsnr_max - logsnr_min)
    
        b = tf.math.atan(tf.exp(-0.5 * logsnr_max))
        a = tf.math.atan(tf.exp(-0.5 * logsnr_min)) - b
        return -2. * tf.math.log(tf.math.tan(a * tf.cast(t,tf.float32) + b))
    
    @tf.function
    def get_logsnr_alpha_sigma(self,time,shape=None):
        logsnr = self.logsnr_schedule_cosine(time,logsnr_min=self.minlogsnr, logsnr_max=self.maxlogsnr)
        alpha = tf.sqrt(tf.math.sigmoid(logsnr))
        sigma = tf.sqrt(tf.math.sigmoid(-logsnr))
        if shape is not None:
            alpha = tf.reshape(alpha,shape)
            sigma = tf.reshape(sigma,shape)
        return logsnr, alpha, sigma

    @tf.function
    def inv_logsnr_schedule_cosine(self,logsnr, logsnr_min, logsnr_max):
        #if self.ll_training:return (logsnr_max - logsnr)/(logsnr_max - logsnr_min)
        b = tf.math.atan(tf.exp(-0.5 * logsnr_max))
        a = tf.math.atan(tf.exp(-0.5 * logsnr_min)) - b
        return tf.math.atan(tf.exp(-0.5 * tf.cast(logsnr,tf.float32)))/a -b/a

        
    @tf.function
    def get_sde(self,time,shape=None):

        with tf.GradientTape(persistent=True,
                             watch_accessed_variables=False) as tape2:
            tape2.watch(time)
            logsnr,alpha,sigma = self.get_logsnr_alpha_sigma(time)
            logalpha= tf.math.log(alpha)
            

        f = tape2.gradient(logalpha, time)
        g2 = -sigma**2*tf.cast(tape2.gradient(logsnr, time),tf.float32)

        if shape is None:
            shape=self.shape
        f = tf.reshape(f,shape)
        g2 = tf.reshape(g2,shape)
        return tf.cast(f,tf.float32), tf.cast(g2,tf.float32)

    

    def marginal_prob(self,t,shape=None):
        logsnr, mean, sigma = self.get_logsnr_alpha_sigma(t)
        if shape is None:
            shape=self.shape
        mean = tf.reshape(mean,shape)
        sigma = tf.reshape(sigma,shape)
        return mean, sigma

    def sde(self,t,shape=None):
        logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(t)
        drift = -0.5*np.pi*sigma/alpha
        diffusion2 = np.pi*sigma/alpha
        if shape is None:
            shape=self.shape
        drift = tf.reshape(drift,shape)
        diffusion2 = tf.reshape(diffusion2,shape)

        return tf.cast(drift,tf.float32), tf.cast(diffusion2,tf.float32)


    @tf.function
    def eval_model(self,model,x,t,jet=None,mask=None):
        if jet is None:
            score = model([x, t])
        else:
            score = model([self.Featurizer(x)*mask, t,jet,mask])*mask
        return score
    
    
    @tf.function
    def train_step(self, inputs):
        eps=1e-5        
        part,jet,mask = inputs
            
        random_t = tf.random.uniform((tf.shape(jet)[0],1))
        
        if self.ll_training:
            #Uniform sampling in logSNR
            random_t = self.inv_logsnr_schedule_cosine(
                (self.maxlogsnr -self.minlogsnr)*random_t + self.minlogsnr,
                logsnr_min=self.minlogsnr, logsnr_max=self.maxlogsnr
            )

            
            
        logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(random_t)
        alpha_reshape = tf.reshape(alpha,self.shape)
        sigma_reshape = tf.reshape(sigma,self.shape)


        with tf.GradientTape() as tape:
            #part
            z = tf.random.normal((tf.shape(part)),dtype=tf.float32)                    
            perturbed_x = alpha_reshape*part + z * sigma_reshape
            pred = self.model_part([self.Featurizer(perturbed_x)*mask,
                                    (logsnr-self.minlogsnr)/(self.maxlogsnr -self.minlogsnr),jet,mask])
            
            if self.ll_training:
                noise = sigma_reshape * perturbed_x + alpha_reshape * pred
                losses = tf.square(z - noise)*mask
            else:
                v = alpha_reshape * z - sigma_reshape * part            
                losses = tf.square(pred - v)*mask
                
            loss_part = tf.reduce_mean(tf.reshape(losses,(tf.shape(losses)[0], -1)))
            
        trainable_variables = self.model_part.trainable_variables
        tape = hvd.DistributedGradientTape(tape) 
        g = tape.gradient(loss_part, trainable_variables)
        g = [tf.clip_by_norm(grad, 1)
             for grad in g]

        self.optimizer.apply_gradients(zip(g, trainable_variables))

        for weight, ema_weight in zip(self.model_part.weights, self.ema_part.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)


        with tf.GradientTape() as tape:
            #jet
            z = tf.random.normal((tf.shape(jet)),dtype=tf.float32)
            
            perturbed_x = alpha*jet + z * sigma
            pred = self.model_jet([perturbed_x,
                                   (logsnr-self.minlogsnr)/(self.maxlogsnr -self.minlogsnr)])
            if self.ll_training:
                noise = sigma * perturbed_x + alpha * pred
                losses = tf.square(z - noise)
            else:
                v = alpha* z - sigma* jet
                losses = tf.square(pred - v)
                
            loss_jet = tf.reduce_mean(tf.reshape(losses,(tf.shape(losses)[0], -1)))
 
        trainable_variables = self.model_jet.trainable_variables
        tape = hvd.DistributedGradientTape(tape) 
        g = tape.gradient(loss_jet, trainable_variables)
        g = [tf.clip_by_norm(grad, 1)
             for grad in g]

        self.optimizer.apply_gradients(zip(g, trainable_variables))        
        self.loss_tracker.update_state(loss_jet + loss_part)

            
        for weight, ema_weight in zip(self.model_jet.weights, self.ema_jet.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)


        return {
            "loss": self.loss_tracker.result(), 
            "loss_part":tf.reduce_mean(loss_part),
            "loss_jet":tf.reduce_mean(loss_jet),
        }


    @tf.function
    def test_step(self, inputs):
        eps = 1e-5
        part,jet,mask = inputs
        
        random_t = tf.random.uniform((tf.shape(jet)[0],1))
        
        if self.ll_training:
            random_t = self.inv_logsnr_schedule_cosine(
                (self.maxlogsnr -self.minlogsnr)*random_t + self.minlogsnr,
                logsnr_min=self.minlogsnr, logsnr_max=self.maxlogsnr
            )

        
        logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(random_t)
        alpha_reshape = tf.reshape(alpha,self.shape)
        sigma_reshape = tf.reshape(sigma,self.shape)
 
        #part
        z = tf.random.normal((tf.shape(part)),dtype=tf.float32)
        perturbed_x = alpha_reshape*part + z * sigma_reshape
        pred = self.model_part([self.Featurizer(perturbed_x)*mask,
                                (logsnr-self.minlogsnr)/(self.maxlogsnr -self.minlogsnr),jet,mask])
        if self.ll_training:
            noise = sigma_reshape * perturbed_x + alpha_reshape * pred
            losses = tf.square(z - noise)*mask
        else:
            v = alpha_reshape * z - sigma_reshape * part            
            losses = tf.square(pred - v)*mask
                        
        loss_part = tf.reduce_mean(tf.reshape(losses,(tf.shape(losses)[0], -1)))
                    
        #jet
        z = tf.random.normal((tf.shape(jet)),dtype=tf.float32)
        
        perturbed_x = alpha*jet + z * sigma
        pred = self.model_jet([perturbed_x,
                               (logsnr-self.minlogsnr)/(self.maxlogsnr -self.minlogsnr)])
        if self.ll_training:
            noise = sigma * perturbed_x + alpha * pred
            losses = tf.square(z - noise)
        else:
            v = alpha* z - sigma* jet
            losses = tf.square(pred- v)
        
        loss_jet = tf.reduce_mean(tf.reshape(losses,(tf.shape(losses)[0], -1)))
        self.loss_tracker.update_state(loss_jet + loss_part)
        
        return {
            "loss": self.loss_tracker.result(), 
            "loss_part":tf.reduce_mean(loss_part),
            "loss_jet":tf.reduce_mean(loss_jet),
        }

            
    @tf.function
    def call(self,x):        
        return self.model(x)


    def get_likelihood(self,part,jet,mask):
        start = time.time()
        ll_jet = self.Likelihood(jet,self.ema_jet)
        ll_part = self.Likelihood(part,self.ema_part,jet=jet,mask=mask)
        return ll_part, ll_jet




    def Likelihood(self,
                   sample,
                   model,
                   jet=None,
                   mask=None,
                   atol=5e-5,
                   eps=1e-4,
                   exact = True,
    ):

        from scipy import integrate

        gc.collect()
        batch_size = sample.shape[0]        
        shape = sample.shape

        
        if mask is None:
            N = np.prod(shape[1:])
            const_shape = (-1,1)
        else:
            N = np.sum(self.num_feat*mask,(1,2))
            const_shape = self.shape

            
        def prior_likelihood(z):
            """The likelihood of a Gaussian distribution with mean zero and 
            standard deviation sigma."""
            shape = z.shape            
            return -N / 2. * np.log(2*np.pi) - np.sum(z.reshape((shape[0],-1))**2, -1) / 2. 

        
        @tf.function
        def divergence_eval_wrapper(sample, time_steps,
                                    jet=None,mask=None):
            
            sample = tf.cast(tf.reshape(sample,shape),tf.float32)
            time_steps = tf.reshape(time_steps,(sample.shape[0], 1))
            time_steps = self.inv_logsnr_schedule_cosine(2*time_steps,
                                                         logsnr_min=self.minlogsnr,
                                                         logsnr_max=self.maxlogsnr
                                                         )
            
            logsnr_steps, alpha, sigma = self.get_logsnr_alpha_sigma(time_steps,shape=const_shape)
            epsilons = tfp.random.rademacher(sample.shape,dtype=tf.float32)
            #epsilons = tf.random.normal(sample.shape,dtype=tf.float32)            
            if mask is not None:
                sample*=mask
                epsilons*=mask

            if exact:
                # Exact trace estimation
                fn = lambda x: -sigma*alpha*self.eval_model(model, x,
                            (logsnr_steps-self.minlogsnr)/(self.maxlogsnr -self.minlogsnr),jet,mask)
            

                pred, diag_jac = diag_jacobian(
                xs=sample, fn=fn, sample_shape=[batch_size])

                if isinstance(pred, list):
                    pred = pred[0]
                    if isinstance(diag_jac, list):
                        diag_jac = diag_jac[0]
            
                return tf.reshape(pred,[-1]), - tf.reduce_sum(tf.reshape(diag_jac,(batch_size,-1)), -1)
            else:
                
                with tf.GradientTape(persistent=False,
                                     watch_accessed_variables=False) as tape:
                    tape.watch(sample)
                    pred = self.eval_model(model,sample,
                                           (logsnr_steps-self.minlogsnr)/(self.maxlogsnr -self.minlogsnr),
                                           jet,mask)                
                    drift = -sigma*alpha*pred
                
                jvp = tf.cast(tape.gradient(drift, sample,epsilons),tf.float32)            
                return  tf.reshape(drift,[-1]), - tf.reduce_sum(tf.reshape(jvp*epsilons,(batch_size,-1)), -1)



        def ode_func(t, x):        
            """The ODE function for use by the ODE solver."""
            time_steps = np.ones((batch_size,)) * t    
            sample = x[:-batch_size]
            logp = x[-batch_size:]
            sample_grad, logp_grad = divergence_eval_wrapper(sample, time_steps,jet,mask)
            return np.concatenate([sample_grad, logp_grad], axis=0)
    
        init_x = np.concatenate([sample.reshape([-1]),np.zeros((batch_size,))],0)
        res = integrate.solve_ivp(
            ode_func,
            #(eps,1.0-eps),
            (self.maxlogsnr/2.0 - eps,self.minlogsnr/2.0+eps),
            init_x,
            #max_step=5e-3,
            rtol=5e-4, atol=atol, method='RK23')
    
        zp = res.y[:, -1]
        z = zp[:-batch_size].reshape(shape)
        if mask is not None:
            z *= mask
            
        delta_logp = zp[-batch_size:].reshape(batch_size)
        prior_logp = prior_likelihood(z)
        return (prior_logp - delta_logp)
