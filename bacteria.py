import numpy as np
import numpy.random as npr
import random
import matplotlib.pyplot as plt
import scipy.stats as ss
from rtnorm import rtnorm

from PIL import Image
import array
import math

import time

class Timer(object):
   def __init__(self, message="elapsed", verbose=True):
      self._verbose=verbose
      self._message=message

   def __enter__(self):
      self._start = time.time()
      return self

   def __exit__(self, *args):
      self._end=time.time()
      self._secs=self._end-self._start
      if self._verbose:
         print "{0} time: {1} ms".format(self.message,self.secs*1000)

class Stopwatch(object):
   def __init__(self, message="elapsed", verbose=True):
      self._verbose=verbose
      self._message=message
      self._secs=0

   def start(self):
      self._start=time.time()
   def stop(self):
      self._secs+=time.time()-self._start
   def reset(self):
      self._secs=0
   def ms(self):
      return self._secs*1000
   def s(self):
      return self._secs
   def report(self):
      if self._verbose:
         print "{0} time: {1} ms".format(self._message,self.ms())

def squash_8bit(val):
   return min(max(val,0),255)

def make_rgb_gradient(start,end):
   def rgb_gradient(alpha):
      alpha=alpha**0.5
      grad=lambda x : squash_8bit((1.0-alpha)*x[0]+alpha*x[1])
      return [int(grad(c)) for c in zip(start,end)]

   return rgb_gradient

def make_color_gradient(start,end):
   def color_gradient(alpha):
      alpha=alpha**0.5
      return int(squash_8bit((1.0-alpha)*start+alpha*end))
   return color_gradient

def generate_image(life_density,death_density,image_filename,verbose_perf):
   # assert life_density.shape==death_density.shape
   bg=[0,0,0]
   max_life_color=[201,53,65]
   max_death_color=[24,154,221]
   rgb_gradient=make_rgb_gradient(bg,max_life_color)
   r_gradient=make_color_gradient(bg[0],max_life_color[0])
   g_gradient=make_color_gradient(bg[1],max_life_color[1])
   b_gradient=make_color_gradient(bg[2],max_life_color[2])
   h,w=life_density.shape
   pixel_list=[0,0,0]*h*w

   with Timer("  writing gradients to pixels",verbose_perf) as t:
      for i in range(h):
         for j in range(w):
            pixel_list[3*(i*w+j)]=r_gradient(life_density[i,j])
            pixel_list[3*(i*w+j)+1]=g_gradient(life_density[i,j])
            pixel_list[3*(i*w+j)+2]=b_gradient(life_density[i,j])
            # pixel_list[3*(i*w+j):3*(i*w+j+1)]=rgb_gradient(density[i,j])

   with Timer("  writing png",verbose_perf) as t:
      img=Image.frombytes('RGB',(h,w),array.array('B',pixel_list))
      img.save(image_filename)

def real_loc_to_pixel_loc(x,y,h,w,xmin,xmax,ymin,ymax):
   assert len(x)==len(y)
   pixel_x=np.fmin((x-xmin)/(xmax-xmin)*w,w-1)
   pixel_y=np.fmin((y-ymin)/(ymax-ymin)*h,h-1)
   return pixel_x.astype(int),pixel_y.astype(int)

def measure_density(x,y,h,w):
   assert len(x)==len(y)
   density=np.zeros((h,w))
   max_density=0

   for i in range(len(x)):
      density[x[i],y[i]]+=1
      max_density=max(density[x[i],y[i]],max_density)

   density/=max_density
   return density

def kldivergence(p,q):
   '''
   KL divergence between two multivariate_normal distributions, in nats
   '''
   detPcov=np.linalg.det(p.cov)
   detQcov=np.linalg.det(q.cov)
   invQcov=np.linalg.inv(q.cov)
   meanDiff=np.matrix((q.mean-p.mean).reshape(-1,1))
   res=0.5*(np.trace(invQcov*p.cov)+meanDiff.T*invQcov*meanDiff-2+math.log(detQcov/detPcov))
   return res[0][0]

def generic_sigm(x,factor):
   return 2.0/(1.0+math.exp(-factor*x))-1.0

class Colony(object):
   def __init__(self,num_bacteria,ident,num_iterations=10,image_size=(400,400),\
         bounds=(-10,10,-10,10),verbose_perf=True):
      self._num_bacteria=int(num_bacteria)
      self._identifier=ident
      self._num_iterations=num_iterations
      self._ident=ident
      self._save_filename="colony_{0}_{1}_n{2}.png"
      self._image_size=image_size
      self._bounds=bounds
      self._verbose_perf=verbose_perf

      # constants
      self._max_num_food_mixture_components=50
      self._food_mixture_components_kl_divergence_threshold=5.0
      self._empty_stomach_penalty=0.2
      self._hunger_wander_factor=4.0
      self._hunger_spread_factor=8.0
      self._eat_food_stdev=0.01
      self._eat_food_weight=0.1/self._num_bacteria
      self._base_food_weight=1.0
      self._hunger_death_factor=1.0

      self._loc_mean=npr.randn(1,2)[0]
      self._loc_cov=ss.wishart.rvs(df=2,scale=np.eye(2)*0.2,size=1)

      self._spread_base_step_size=0.5
      self._wander_base_step_size=0.05
      self._wander_jump_size=3.0

      self._base_food_mean=npr.rand(1,2)[0]*2
      self._base_food_cov=ss.wishart.rvs(df=2,scale=np.eye(2)*10,size=1)

      # generally, reproduce factor should be higher than death factor, which will cause the
      # microbes to reproduce earlier than they die
      # these could be randomly generated?
      self._age_death_factor=0.25
      self._age_reproduce_factor=0.5

   def grow(self):
      self.init()
      for i in range(self._num_iterations):
         print "Iteration {0} of {1}".format(i+1,self._num_iterations)
         with Timer("step time",self._verbose_perf) as t:
            self.step(i+1)

   def init(self):

      with Timer("generating random bacteria",self._verbose_perf) as t:
         with Timer(" preparing microbes",self._verbose_perf) as t2:
            self.prepare_microbes()
         with Timer(" preparing environment",self._verbose_perf) as t2:
            self.prepare_environment()

      with Timer("creating image",self._verbose_perf) as t:
         self.save_image("init")

   def prepare_microbes(self):
      with Timer("  generating locations",self._verbose_perf) as t:
         self.generate_locations()
      with Timer("  generating traits",self._verbose_perf) as t:
         self.generate_traits()
      with Timer("  generating statuses",self._verbose_perf) as t:
         self.generate_statuses()

   def generate_locations(self):
      self._x,self._y=npr.multivariate_normal(self._loc_mean,self._loc_cov,self._num_bacteria).T
      self._deaths_x=np.ones((0,))
      self._deaths_y=np.ones((0,))

   def generate_traits(self):
      self.generate_resistance()
      self.generate_heat_tolerance()
      self.generate_cold_tolerance()
      self.generate_competence()

   def generate_resistance(self):
      pass
   def generate_heat_tolerance(self):
      pass
   def generate_cold_tolerance(self):
      pass
   def generate_competence(self):
      pass

   def generate_statuses(self):
      self.generate_hunger()
      self.generate_age()

   def generate_hunger(self):
      self._hunger=np.zeros((self._num_bacteria,))
   def generate_age(self):
      self._age=np.zeros((self._num_bacteria,))

   def prepare_environment(self):
      self.prepare_food()
      self.initialize_temperature()
      self.initialize_antibiotics()

   def prepare_food(self):
      # food is a psuedo-probability represented as a mixture of gaussians
      self._base_food_dist=ss.multivariate_normal(self._base_food_mean,self._base_food_cov)
      self._eaten_food_dists=[]
      self._eaten_food_weights=[]

   def food_prob(self,x):
      assert len(self._eaten_food_dists)==len(self._eaten_food_weights)
      p=self._base_food_weight*np.ones(x.shape[:-1])
      #p=self._base_food_weight*self._base_food_dist.pdf(x)
      for i in range(len(self._eaten_food_dists)):
         p-=self._eaten_food_weights[i]*self._eaten_food_dists[i].pdf(x)
      return p

   def initialize_temperature(self):
      pass
   def initialize_antibiotics(self):
      pass

   def save_image(self,status):
      xmin,xmax,ymin,ymax=self._bounds
      h,w=self._image_size
      with Timer(" converting real number space to pixel space",self._verbose_perf) as t:
         pixel_x,pixel_y=real_loc_to_pixel_loc(self._x,self._y,h,w,xmin,xmax,ymin,ymax)
      with Timer(" measuring density",self._verbose_perf) as t:
         density=measure_density(pixel_x,pixel_y,h,w)
      with Timer(" generating image",self._verbose_perf) as t:
         filename=self._save_filename.format(self._ident,status,self._num_bacteria)
         generate_image(density,None,filename,self._verbose_perf)

   def save_food_image(self,status):
      xmin,xmax,ymin,ymax=self._bounds
      h,w=self._image_size
      x,y=np.mgrid[-10:10:0.1,-10:10:0.1]
      pos=np.dstack((x,-y))
      plt.contourf(x,y,self.food_prob(pos))
      plt.axis('off')
      plt.savefig(self._save_filename.format(self._ident,status,self._num_bacteria),\
         bbox_inches='tight')

   def step(self,step_num):
      with Timer(" updating environment",self._verbose_perf) as t:
         self.update_environment()
      with Timer(" updating microbes",self._verbose_perf) as t:
         self.update_microbes()

      with Timer("creating image",self._verbose_perf) as t:
         self.save_image("step{0}".format(step_num))
         self.save_food_image("step{0}_food".format(step_num))

   def update_environment(self):
      self.update_food()
      self.update_temperature()
      self.update_antibiotics()

   def update_food(self):
      # consolidate extraneous distributions
      # n^2 in self._max_num_food_mixture_components
      merge=[-1]*len(self._eaten_food_dists)
      for i in range(len(self._eaten_food_dists)):
         if merge[i]>-1:
            continue
         for j in range(len(self._eaten_food_dists)):
            if j>i and merge[j]==-1:
               klij=kldivergence(self._eaten_food_dists[i],self._eaten_food_dists[j])
               klji=kldivergence(self._eaten_food_dists[j],self._eaten_food_dists[i])
               if klij<self._food_mixture_components_kl_divergence_threshold and klij<klji:
                  # using j to approximate i loses more information than using i to approximate
                  # j, so merge j into i
                  merge[j]=i
               elif klji<self._food_mixture_components_kl_divergence_threshold and klji<klij:
                  # using i to approximate j loses more information than using j to approximate i,
                  # so merge i in to j
                  merge[i]=j

      new_size=np.sum(np.array(merge)==-1)
      new_eaten_food_dists=[None]*new_size
      new_eaten_food_weights=[0.0]*new_size
      i=0
      for j in range(len(merge)):
         if merge[j]==-1:
            new_eaten_food_dists[i]=self._eaten_food_dists[j]
            new_eaten_food_weights[i]+=self._eaten_food_weights[j]
            i+=1
         else:
            new_eaten_food_weights[merge[j]]+=self._eaten_food_weights[j]
            new_eaten_food_dists[i].cov+=self._eaten_food_dists[j].cov
      print "num_merged_eaten_dists={0}".format(len(merge)-i)

      self._eaten_food_dists=new_eaten_food_dists
      self._eaten_food_weights=new_eaten_food_weights

   def update_temperature(self):
      pass
   def update_antibiotics(self):
      pass

   def update_microbes(self):
      with Timer(" microbes eating",self._verbose_perf) as t:
         num_found_food=self.eat()
         print "num_found_food={0} pct_found_food={1} avg_hunger={2}".format(\
            num_found_food,float(num_found_food)/self._num_bacteria,np.mean(self._hunger))
         print "num_eaten_food_dists={0} total_eaten_food_weights={1}".format(\
            len(self._eaten_food_dists),np.sum(self._eaten_food_weights))
      with Timer(" microbes foraging",self._verbose_perf) as t:
         num_spread,avg_dist_spread,num_wandered=self.forage()
         pct_spread=0.0
         pct_wandered=0.0
         if self._num_bacteria > 0:
            pct_spread=float(num_spread)/self._num_bacteria
            pct_wandered=float(num_wandered)/self._num_bacteria
         print "num_spread={0} pct_spread={1} avg_dist_spread={2} num_wandered={3}"
            " pct_wandered={4}".format(num_spread,pct_spread,avg_dist_spread,num_wandered,\
            pct_wandered)
      with Timer(" microbes transferring",self._verbose_perf) as t:
         self.transfer()
      with Timer(" microbes reproducing and dying",self._verbose_perf) as t:
         self.reproduce_or_die()

   def eat(self):
      sw1=Stopwatch("**food_prob",self._verbose_perf)
      sw2=Stopwatch("**eat_food_at",self._verbose_perf)
      sw_make_dist=Stopwatch("**making eating dist",self._verbose_perf)
      sw_compute_kl=Stopwatch("**computing KL divergence",self._verbose_perf)
      sw_main=Stopwatch("*main",self._verbose_perf)
      sw_main.start()
      sw1.start()
      food_probs=self.food_prob(np.array([self._x,self._y]).T)
      if np.isscalar(food_probs):
         food_probs=np.array([food_probs])
      sw1.stop()
      # sw1.report()
      num_found_food=0
      for i in range(self._num_bacteria):
         if i>0 and i % 10000==0:
            sw_main.stop(); sw_main.report(); sw_main.reset(); sw_main.start()
            sw2.report(); sw2.reset()
            sw_make_dist.report(); sw_make_dist.reset()
            sw_compute_kl.report(); sw_compute_kl.reset()
         food_roll=random.uniform(0,1)
         found_food=food_roll<food_probs[i]
         # print food_roll,food_probs[i],found_food
         if found_food:
            # food found, eat it!
            sw2.start()
            self.eat_food_at([self._x[i],self._y[i]],sw_make_dist,sw_compute_kl)
            sw2.stop()
            self._hunger[i]=0.0
            num_found_food+=1
         else:
            self._hunger[i]+=self._empty_stomach_penalty

      return num_found_food

   def eat_food_at(self,loc,sw_make_dist,sw_compute_kl):
      sw_make_dist.start()
      eating_dist=ss.multivariate_normal(loc,[[self._eat_food_stdev,0],[0,self._eat_food_stdev]])
      sw_make_dist.stop()
      min_divergence=float("inf")
      min_div_idx=-1
      added_to_existing=False
      for i in range(len(self._eaten_food_dists)):
         sw_compute_kl.start()
         kldiv=kldivergence(eating_dist,self._eaten_food_dists[i])
         sw_compute_kl.stop()
         if kldiv<min_divergence:
            min_divergence=kldiv
            min_div_idx=i
         if kldiv < self._food_mixture_components_kl_divergence_threshold:
            self._eaten_food_weights[i]+=self._eat_food_weight
            added_to_existing=True

      if not added_to_existing:
         if len(self._eaten_food_dists)<self._max_num_food_mixture_components:
            self._eaten_food_dists.append(eating_dist)
            self._eaten_food_weights.append(self._eat_food_weight)
         else:
            self._eaten_food_weights[min_div_idx]+=self._eat_food_weight

   def forage(self):
      sum_spread_distance=0.0
      num_spread=0
      num_wandered=0
      for i in range(self._num_bacteria):
         hunger_roll=random.uniform(0,1)
         if hunger_roll<self.hunger_wander_sigm(self._hunger[i]):
            # we're hungry, go wandering!
            before_x=self._x[i]
            before_y=self._y[i]
            self._x[i],self._y[i]=self.wander(self._x[i],self._y[i],self._motility[i])
            num_wandered+=1
         elif hunger_roll<self.hunger_spread_sigm(self._hunger[i]):
            # we're kinda hungry, wriggle around a little bit
            before_x=self._x[i]
            before_y=self._y[i]
            self._x[i],self._y[i],dist=self.move(self._x[i],self._y[i],self._motility[i])
            # print "before=({0},{1}) after=({2},{3}) dist={4} actual_dist={5}".format(\
            #    before_x,before_y,self._x[i],self._y[i],dist,
            #    math.sqrt((self._x[i]-before_x)**2+(self._y[i]-before_y)**2))
            sum_moved_distance+=dist
            num_spread+=1

      avg_spread_distance=0.0
      if num_spread>0:
         avg_spread_distance=sum_spread_distance/num_spread
      return num_spread,avg_spread_distance,num_wandered

   def move_random(self,x,y):
      new_x,new_y=npr.multivariate_normal([0,0],np.eye(2),1)[0]*self._motility_base_step_size
      return new_x,new_y

   def spread(self,x,y,motility):
      dist=random.uniform(0.0,motility*self._spread_base_step_size)
      # print motility,self._motility_base_step_size,dist,theta,dist*math.sin(theta),dist*math.cos(theta)
      new_x,new_y=self.move_random_direction(x,y,dist)
      return new_x,new_y,dist

   def move_random_direction(self,x,y,dist):
      theta=random.uniform(0.0,2*math.pi)
      return move(x,y,dist,theta):

   def move(self,x,y,dist,theta):
      return x+dist*math.cos(theta),y+dist*math.sin(theta)

   def wander(self,x,y):
      # jump to random spot
      wander_x,wander_y=move_random_direction(x,y,self._wander_jump_size)
      neighbors=self._kd_tree.query_ball_point([x,y],self._bacteria_radius)
      while len(neighbors)==0:
         dist=random.uniform(0.0,self._wander_base_step_size)
         wander_x,wander_y=


   def hunger_wander_sigm(self,x):
      return generic_sigm(x,self._hunger_wander_factor)
   def hunger_spread_sigm(self,x):
      return generic_sigm(x,self._hunger_spread_factor)

   def transfer(self):
      pass

   def reproduce_or_die(self):
      #everyone gets one iteration older
      self._age+=1.0
      deaths=np.zeros((self._num_bacteria,))
      reproducing=np.zeros((self._num_bacteria,))
      for i in range(self._num_bacteria):
         if random.uniform(0,1)<self.death_sigm(self._age[i]) or \
               random.uniform(0,1)<self.hunger_death_sigm(self._hunger[i]):
            # die!
            deaths[i]=1.0
         elif random.uniform(0,1)<self.reproduce_sigm(self._age[i]):
            # reproduce (by fission)!
            reproducing[i]=1.0

      num_reproductions=int(np.sum(reproducing))
      num_deaths=int(np.sum(deaths))
      # new number of bacteria is old count, minus the number of died, plus the number of
      # reproductions that happened (since each producing microbes becomes 2 microbes)
      new_num_bacteria=self._num_bacteria-num_deaths+num_reproductions
      print "num_deaths={0} num_reproductions={1} num_new_microbes={2} "\
            "new_bacteria_count={3}".format(num_deaths,num_reproductions,2*num_reproductions,\
            new_num_bacteria)
      new_x=np.zeros((new_num_bacteria,))
      new_y=np.zeros((new_num_bacteria,))
      additional_deaths_x=np.zeros((num_deaths,))
      additional_deaths_y=np.zeros((num_deaths,))
      new_age=np.zeros((new_num_bacteria,))
      new_hunger=np.zeros((new_num_bacteria,))
      #TODO: add rest of traits
      i=0
      k=0

      def copy_old_into_new(old_idx,new_idx):
         new_x[new_idx]=self._x[old_idx]
         new_y[new_idx]=self._y[old_idx]
         new_age[new_idx]=self._age[old_idx]
         new_hunger[new_idx]=self._hunger[old_idx]

      for j in range(self._num_bacteria):
         if deaths[j]:
            additional_deaths_x[k]=self._x[j]
            additional_deaths_y[k]=self._y[j]
            k+=1
         elif not reproducing[j]:
            assert i<new_num_bacteria
            # copy non-dead, non-reproducing bacteria into new array
            copy_old_into_new(j,i)
            i+=1

      assert k==num_deaths
      # now add new bacteria
      def reproduce(old_idx,new_idx):
         new_x[new_idx]=self._x[old_idx]
         new_y[new_idx]=self._y[old_idx]
         # age stays zero
         new_hunger[new_idx]=self._hunger[old_idx]

      for j in range(self._num_bacteria):
         if reproducing[j]:
            assert i<new_num_bacteria
            reproduce(j,i)
            i+=1
            reproduce(j,i)
            i+=1

      assert i==new_num_bacteria

      # all done, save the new state
      self._num_bacteria=new_num_bacteria
      self._x=new_x
      self._y=new_y
      self._age=new_age
      self._hunger=new_hunger
      self._deaths_x=np.concatenate((self._deaths_x,additional_deaths_x))
      self._deaths_y=np.concatenate((self._deaths_y,additional_deaths_y))

   def death_sigm(self,x):
      return generic_sigm(x,self._age_death_factor)
   def hunger_death_sigm(self,x):
      return generic_sigm(x,self._hunger_death_factor)
   def reproduce_sigm(self,x):
      return generic_sigm(x,self._age_reproduce_factor)

   def reproduce(self):
      pass

if __name__ == "__main__":
   num_colonies=1
   for i in range(num_colonies):
      c=Colony(num_bacteria=1e3,ident=i+1,verbose_perf=False,num_iterations=200)
      c.grow()
