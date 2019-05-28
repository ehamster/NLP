1.学习指数衰减法
====================

exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None)

通常在训练的时候建议学习率衰

<a href="https://www.codecogs.com/eqnedit.php?latex=\fn_phv&space;\LARGE&space;learning\_rate*decay\_rate^{\frac{global\_step}{decay\_steps}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\fn_phv&space;\LARGE&space;learning\_rate*decay\_rate^{\frac{global\_step}{decay\_steps}}" title="\LARGE learning\_rate*decay\_rate^{\frac{global\_step}{decay\_steps}}" /></a>




