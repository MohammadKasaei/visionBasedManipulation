import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from   matplotlib.patches import Rectangle, Ellipse


class visualization():
    def __init__(self,showDotPlots=False) -> None:
        self.showDotPlots = showDotPlots
        


    def init_animation(self,ref,x0,t,ts):
        
            self.ts  = ts
            self.ref = ref

            self.fig = plt.figure(figsize=(5, 4))
            self.ax  = self.fig.add_subplot(211, autoscale_on=False, xlim=(-2, 2), ylim=(-0.2, 0.5))
            self.ax.set_aspect('equal')
            self.ax.set(xlabel='x (m)', ylabel='y (m)')
            

            self.ax.grid()
            self.line, = self.ax.plot([], [], 'o-', color = "red", lw=2)
            self.time_template = 'time = %.1fs'
            self.time_text = self.ax.text(0.05, 0.9, '', transform=self.ax.transAxes)
            
            self.cart_h = 0.1
            self.cart_w = 0.2
            self.cart_rect = Rectangle(([x0[0]-(self.cart_w / 2),0.05]),self.cart_h,self.cart_w, fc="blue", ec="black", lw=1.0)
            self.ax.add_patch(self.cart_rect)

            self.goal_size = 0.05        
            self.goal_rect = Rectangle(([self.ref[0,0]-(self.goal_size / 2),0.0]),self.goal_size,self.goal_size, fc="green", ec="black", lw=1.0)
            self.ax.add_patch(self.goal_rect)

            self.wheel_r = 0.02
            self.cart_wh1 = Ellipse(([x0[0]-(self.cart_w / 2)+0.04,0.05]),self.wheel_r,self.wheel_r, fc="black", ec="black", lw=1.0)
            self.ax.add_patch(self.cart_wh1)

            self.cart_wh2 = Ellipse(([x0[0]-(self.cart_w / 2)+self.cart_w-0.04,0.05]),self.wheel_r,self.wheel_r, fc="black", ec="black", lw=1.0)
            self.ax.add_patch(self.cart_wh2)
            
            self.fig.add_subplot(212)

            plt.plot(t, self.animation_states[0:len(t),0], 'b', label=r'$x(t)$')
            plt.plot(t, self.animation_states[0:len(t),2], 'r', label=r'$\theta(t)$')
            plt.plot(t,ref[0:len(t),0],'g',label='ref')

            if (self.showDotPlots):
                plt.plot(t, self.animation_states[0:len(t),1], 'c', label=r'$\dot{x}(t)$')
                plt.plot(t, self.animation_states[0:len(t),3], 'k', label=r'$\dot{\theta}(t)$')
                plt.plot(t, np.diff(self.animation_states[0:len(t),1],append=0), 'm', label=r'$\ddot{x}(t)$')
                plt.plot(t, np.diff(self.animation_states[0:len(t),3],append=0), 'y', label=r'$\ddot{\theta}(t)$')
                
            plt.legend(loc='best')
            plt.xlabel('t (s)')
            plt.ylabel('pos and angle (m - rad)')
            plt.grid(color='k', ls = '-.', lw = 0.25)

            #plt.show()
            #   
    def show_animation(self,ref,t,ts,x0,x,tfinal,AniPrevTime,maximized):    
            self.animation_states = x
            self.init_animation(ref,x0,t,ts)
            ani = animation.FuncAnimation(self.fig, self.animate, int(len(x)), interval=int(tfinal/ts)/100000, blit=True)
            # ani = animation.FuncAnimation(self.fig, self.animate, int(len(x)), interval=0.07, blit=True)
            
            if (maximized):
                mng = plt.get_current_fig_manager()
                mng.window.showMaximized()
            
            if (AniPrevTime==-1):
                plt.show()
            else:
                plt.show(block=False)
                plt.pause(AniPrevTime)
                plt.close()


    def animate(self,i):
            L = 0.3
            cart_pos = [self.animation_states[i,0],0.1]
            pend_pos = [cart_pos[0]+L*np.sin(self.animation_states[i,2]),cart_pos[1]+L*np.cos(self.animation_states[i,2])]
            x_pos = [cart_pos[0],pend_pos[0]]
            y_pos = [cart_pos[1],pend_pos[1]]
            self.line.set_data(x_pos, y_pos)

            self.cart_rect.set_width(self.cart_w)
            self.cart_rect.set_height(self.cart_h)
            cart_cord = cart_pos-np.array([self.cart_w/2,self.cart_h/2])        
            self.cart_rect.set_xy(cart_cord)
            
            self.cart_wh1.set_center(cart_cord + np.array([0.04,0]))
            self.cart_wh2.set_center(cart_cord + np.array([self.cart_w -0.04,0]))
                    
            self.goal_size = 0.05        
            self.goal_rect.set_width(self.goal_size)
            self.goal_rect.set_height(self.goal_size)
            goal_cord = [self.ref[i,0],0.1]-np.array([self.goal_size/2,self.goal_size/2])        
            self.goal_rect.set_xy(goal_cord)
            

            self.time_text.set_text(self.time_template % (i*self.ts))

            return self.line, self.time_text, self.cart_rect, self.cart_wh1, self.cart_wh2,self.goal_rect