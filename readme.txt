simple kriging interpolator given a power spectrum
and points on a regularly sampled grid.  You specify
a number of neighboring points to use and a fine
sampling of a grid cell.  Call ftkrig.get_coeffs_1d
once to get the weights, then you can repeatedly call
ftkrig.eval_krig to get your function at interpolated
values.
