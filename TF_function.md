1.学习指数衰减法
====================

exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None)

通常在训练的时候建议学习率衰
decayed_learning_rate = learning_rate*decay_rate^{\frac{global_step}{decay_steps}}
<a href="https://www.codecogs.com/eqnedit.php?latex=learning_rate*decay_rate^{\frac{global_step}{decay_steps}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?learning_rate*decay_rate^{\frac{global_step}{decay_steps}}" title="learning_rate*decay_rate^{\frac{global_step}{decay_steps}}" /></a>










<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>
