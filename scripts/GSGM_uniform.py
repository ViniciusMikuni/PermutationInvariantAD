import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
import time
#import horovod.tensorflow.keras as hvd
import utils
import tensorflow_probability as tfp
from deepsets_cond import DeepSetsAtt, Resnet
from tensorflow.keras.activations import swish, relu
import gc
#tf.random.set_seed(1234)

class GSGM(keras.Model):
    """Score based generative model"""
    def __init__(self,name='SGM',npart=30,
                 particle='gluon_tagging',
                 ll_training=False,config=None):
        super(GSGM, self).__init__()

        self.config = config
        if config is None:
            raise ValueError("Config file not given")

        self.activation = layers.LeakyReLU(alpha=0.01)
        
        self.beta_0 = 0.1
        self.beta_1 = 20.0
        
        
        #self.activation = layers.LeakyReLU(alpha=0.01)
        self.num_feat = self.config['NUM_FEAT']
        self.num_jet = self.config['NUM_JET']
        self.num_embed = self.config['EMBED']
        self.max_part = npart
        self.particle = particle
        self.ll_training = ll_training
        self.ema=0.999

        
        #self.verbose = 1 if hvd.rank() == 0 else 0 #show progress only for first rank

        
        self.projection = self.GaussianFourierProjection(scale = 16)
        self.loss_tracker = keras.metrics.Mean(name="loss")


        #Transformation applied to conditional inputs
        inputs_time = Input((1))
        inputs_jet = Input((self.num_jet))
        inputs_mask = Input((None,1)) #mask to identify zero-padded objects
        

        graph_conditional = self.Embedding(inputs_time,self.projection)
        jet_conditional = self.Embedding(inputs_time,self.projection)


        
        graph_conditional = layers.Dense(self.num_embed,activation=None)(tf.concat(
            [graph_conditional,inputs_jet],-1))
        graph_conditional=self.activation(graph_conditional)
        
        jet_conditional = layers.Dense(self.num_embed,activation=None)(jet_conditional)
        jet_conditional=self.activation(jet_conditional)

        
        self.shape = (-1,1,1)
        inputs,outputs = DeepSetsAtt(
            num_feat=self.num_feat,
            time_embedding=graph_conditional,
            num_heads=1,
            num_transformer = 8,
            projection_dim = 64,
            mask = inputs_mask,
            fourier_features = False
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
            fourier_features = False
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
        angle = inputs*projection
        embedding = tf.concat([tf.math.sin(angle),tf.math.cos(angle)],-1)
        # embedding = layers.Dense(2*self.num_embed,activation=None)(embedding)
        # embedding = self.activation(embedding)
        # embedding = layers.Dense(self.num_embed)(embedding)
        return embedding

    def prior_sde(self,dimensions):
        return tf.random.normal(dimensions,dtype=tf.float32)

    def marginal_prob(self,t,shape=None):        
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0        
        if shape is None:
            shape=self.shape
        log_mean_coeff = tf.reshape(log_mean_coeff,shape)
        mean = tf.exp(log_mean_coeff)
        std = tf.math.sqrt(1 - tf.exp(2. * log_mean_coeff))
        return mean, std

    def sde(self,t,shape=None):        
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        if shape is None:
            shape=self.shape
        beta_t = tf.reshape(beta_t,shape)
        drift = -0.5 * beta_t
        diffusion = tf.math.sqrt(beta_t)

        return tf.cast(drift,tf.float32), tf.cast(diffusion,tf.float32)



    def likelihood_importance_cum_weight(self, t, eps=1e-5):
        #VPSDE
        exponent1 = 0.5 * eps * (eps - 2) * self.beta_0 - 0.5 * eps ** 2 * self.beta_1
        exponent2 = 0.5 * t * (t - 2) * self.beta_0 - 0.5 * t ** 2 * self.beta_1
        term1 = tf.where(tf.abs(exponent1) <= 1e-3, -exponent1, 1. - tf.exp(exponent1))
        term2 = tf.where(tf.abs(exponent2) <= 1e-3, -exponent2, 1. - tf.exp(exponent2))
        return 0.5 * (-2 * tf.math.log(term1) + 2 * tf.math.log(term2)
                      + self.beta_0 * (-2 * eps + eps ** 2 - (t - 2) * t)
                      + self.beta_1 * (-eps ** 2 + t ** 2))

    def sample_importance_weighted_time_for_likelihood(self, shape, quantile=None, eps=1e-5, steps=100):
        Z = self.likelihood_importance_cum_weight(1.0, eps=eps)
        if quantile is None:
            quantile = tf.random.uniform(shape, minval=0, maxval=Z, dtype=tf.float32)
        lb = tf.ones_like(quantile) * eps
        ub = tf.ones_like(quantile) * 1.0

        def bisection_func(carry, idx):
            lb, ub = carry
            lb = tf.cast(lb,tf.float32)
            ub = tf.cast(ub,tf.float32)
            mid = (lb + ub) / 2.0
            value = self.likelihood_importance_cum_weight(mid, eps=eps)
            lb = tf.where(value <= quantile, mid, lb)
            ub = tf.where(value <= quantile, ub, mid)
            return (lb, ub), idx

        for i in tf.range(0,steps):
            (lb, ub), _ = bisection_func((lb, ub), i)
        
        return (lb + ub) / 2.0


    
    
    @tf.function
    def train_step(self, inputs):
        eps=1e-5        
        part,jet,mask = inputs
        
        random_t = tf.random.uniform((tf.shape(jet)[0],1))*(1-eps) + eps
        if self.ll_training:
            random_t = self.sample_importance_weighted_time_for_likelihood((tf.shape(part)[0],1), eps=eps)
            
             
             
        mean,std = self.marginal_prob(random_t)
        with tf.GradientTape() as tape:
            #part
            z = tf.random.normal((tf.shape(part)),dtype=tf.float32)
                    
            perturbed_x = mean*part + z * std 
            score = self.model_part([perturbed_x*mask, random_t,jet,mask])
            
            losses = tf.square(score*std + z)*mask
            loss_part = tf.reduce_mean(tf.reshape(losses,(tf.shape(losses)[0], -1)))
            
        trainable_variables = self.model_part.trainable_variables
        g = tape.gradient(loss_part, trainable_variables)
        # g = [tf.clip_by_norm(grad, 1)
        #      for grad in g]

        self.optimizer.apply_gradients(zip(g, trainable_variables))

        for weight, ema_weight in zip(self.model_part.weights, self.ema_part.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)
        
        with tf.GradientTape() as tape:
            #jet
            z = tf.random.normal((tf.shape(jet)),dtype=tf.float32)
            #mean,std = self.marginal_prob(random_t,shape=(-1,1))
            
            perturbed_x = tf.reshape(mean,(-1,1))*jet + z * tf.reshape(std,(-1,1))
            score = self.model_jet([perturbed_x, random_t])
            losses = tf.square(score*tf.reshape(std,(-1,1)) + z)
            loss_jet = tf.reduce_mean(tf.reshape(losses,(tf.shape(losses)[0], -1)))
 
        trainable_variables = self.model_jet.trainable_variables
        g = tape.gradient(loss_jet, trainable_variables)
        # g = [tf.clip_by_norm(grad, 1)
        #      for grad in g]

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

        random_t = tf.random.uniform((tf.shape(jet)[0],1))*(1-eps) + eps

        if self.ll_training:
            random_t = self.sample_importance_weighted_time_for_likelihood((tf.shape(part)[0],1), eps=eps)                    
        mean,std = self.marginal_prob(random_t)
        #part
        z = tf.random.normal((tf.shape(part)),dtype=tf.float32)
        perturbed_x = mean*part + z * std 
        score = self.model_part([perturbed_x, random_t,jet,mask])            
        losses = tf.square(score*std + z)*mask
                        
        loss_part = tf.reduce_mean(tf.reshape(losses,(tf.shape(losses)[0], -1)))
                    
        #jet
        z = tf.random.normal((tf.shape(jet)),dtype=tf.float32)
        mean,std = self.marginal_prob(random_t,shape=(-1,1))
        perturbed_x = mean*jet + z * std
        score = self.model_jet([perturbed_x, random_t])
        losses = tf.square(score*std + z)
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
        ll_jet = self.Likelihood(jet,self.model_jet)
        #ll_part = self.Likelihood(part,self.model_part,jet=jet,mask=mask)
        ll_part = ll_jet
        #ll_jet = ll_part
        return ll_jet + ll_part




    def Likelihood(self,
                   sample,
                   model,
                   jet=None,
                   mask=None,
                   atol=1e-5,
                   eps=1e-5
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
        

        #epsilons = tfp.random.rademacher(sample.shape,dtype=tf.float32)
        #epsilons = tf.random.normal(sample.shape,dtype=tf.float32)
        
        @tf.function
        def divergence_eval_wrapper(sample, time_steps,
                                    jet=None,mask=None):
            
            sample = tf.cast(tf.reshape(sample,shape),tf.float32)
            f,g = self.sde(time_steps,shape=const_shape)
            #mean,std = self.marginal_prob(time_steps)
            epsilons = tfp.random.rademacher(sample.shape,dtype=tf.float32)
            with tf.GradientTape(persistent=False,
                                 watch_accessed_variables=False) as tape:
                tape.watch(sample)

                if jet is None:
                    score = model([sample, time_steps])
                else:
                    score = model([sample*mask, time_steps,jet,mask])*mask
                        
                drift = f*sample - 0.5*g**2*score
            jvp = tf.cast(tape.gradient(drift, sample,epsilons),tf.float32)
            grads = jvp*epsilons
            if mask is not None:
                grads *= mask
            return  tf.reshape(drift,[-1]), tf.reduce_sum(tf.reshape(grads,(batch_size,-1)), -1)



        def ode_func(t, x):        
            """The ODE function for use by the ODE solver."""
            time_steps = np.ones((batch_size,)) * t    
            sample = x[:-batch_size]
            logp = x[-batch_size:]
            sample_grad, logp_grad = divergence_eval_wrapper(sample, time_steps,jet,mask)
            return np.concatenate([sample_grad, logp_grad], axis=0)
    
        init_x = np.concatenate([sample.reshape([-1]),np.zeros((batch_size,))],0)
        res = integrate.solve_ivp(
            ode_func, (eps,1.0), init_x,
            rtol=1e-4, atol=atol, method='RK45')
    
        zp = res.y[:, -1]
        z = zp[:-batch_size].reshape(shape)
        if mask is not None:
            z *= mask
            
        delta_logp = zp[-batch_size:].reshape(batch_size)
        prior_logp = prior_likelihood(z)
        return (prior_logp + delta_logp)/N

