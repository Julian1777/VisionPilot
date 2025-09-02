import numpy as np


class PIDController:
    def __init__(self, Kp=1.0, Ki=0.1, Kd=0.05, integral_limit=1.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral_limit = integral_limit
        
        self.previous_error = 0.0
        self.integral = 0.0
        
    def update(self, error, dt):
        if dt <= 0:
            dt = 0.01
            
        # Proportional term
        p_term = self.Kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = self.Ki * self.integral
        
        # Derivative term
        derivative = (error - self.previous_error) / dt
        d_term = self.Kd * derivative
        
        # Store for next iteration
        self.previous_error = error
        
        # Calculate output
        output = p_term + i_term + d_term
        return output
    
    def reset(self):
        self.previous_error = 0.0
        self.integral = 0.0
