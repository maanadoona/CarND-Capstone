import rospy

MIN_NUM = float('-inf')
MAX_NUM = float('inf')


class PID(object):
    def __init__(self, kp, ki, kd, mn=MIN_NUM, mx=MAX_NUM):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.diff = 0.0
        self.int_cte = 0.0
        self.prev_cte = 0.0
        self.steer = 0.0

    def reset(self):
        self.int_cte = 0.0
        self.prev_cte = 0.0

    def step(self, error, sample_time):
        self.diff = self.error - self.prev_cte
        self.int_cte += error

        self.steer = -self.kp*error - self.kd*self.diff - self.ki*self.int_cte

        self.prev_cte = error

        if self.steer > 1:
            self.steer = 1

        if self.steer < -1:
            self.steer = -1

        return self.steer
        '''
        integral = self.int_val + error * sample_time;
        derivative = (error - self.last_error) / sample_time if sample_time > 0  else 0;

        y = self.kp * error + self.ki * self.int_val + self.kd * derivative;
        val = max(self.min, min(y, self.max))

        if val > self.max:
            val = self.max
        elif val < self.min:
            val = self.min
        else:
            self.int_val = integral
        self.last_error = error

        return val
        '''