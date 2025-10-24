import numpy as np


class PIDController:
    def __init__(self, Kp=1.0, Ki=0.1, Kd=0.05, integral_limit=1.0, derivative_filter_alpha=0.3):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral_limit = integral_limit
        self.derivative_filter_alpha = derivative_filter_alpha  # Low-pass filter for derivative
        
        self.previous_error = 0.0
        self.integral = 0.0
        self.filtered_derivative = 0.0
        
    def update(self, error, dt):
        if dt <= 0:
            dt = 0.01

        # Reset integral if error sign changes (crosses setpoint)
        if np.sign(error) != np.sign(self.previous_error) and abs(self.previous_error) > 1e-6:
            self.integral = 0.0

        # Proportional term
        p_term = self.Kp * error

        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = self.Ki * self.integral

        # Derivative term with low-pass filtering to prevent spikes
        raw_derivative = (error - self.previous_error) / dt
        # Exponential smoothing: filtered_deriv = alpha * raw + (1-alpha) * prev_filtered
        self.filtered_derivative = (
            self.derivative_filter_alpha * raw_derivative + 
            (1 - self.derivative_filter_alpha) * self.filtered_derivative
        )
        d_term = self.Kd * self.filtered_derivative

        # Store for next iteration
        self.previous_error = error

        # Calculate output
        output = p_term + i_term + d_term
        return output
    
    def reset(self):
        self.previous_error = 0.0
        self.integral = 0.0
        self.filtered_derivative = 0.0
