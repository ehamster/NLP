1.学习指数衰减法
====================

exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None)

通常在训练的时候建议学习率衰减
decayed_learning_rate = learning_rate*$decay_rate^{($\frac{global_setp}{decay_steps}$)}
