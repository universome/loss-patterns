Experiments on mode connectivity and loss patterns

TODO:
* [ ] Profiling / make minimum visualization faster / make training faster
* [ ] It's impossible to make batchnorm/layernorm/actnorm layers orthogonal, because they all have std/scale parameter, which is always greater or equal to zero, so we can't do anything about there orthogonalization, except ignoring it. why do we make scale parameter to be like std? can't we make it any we like, even negative?
* [ ] Plots for paper (averaging over different runs, etc.)
* [ ] CIFAR-10
* [x] Make some cells more important (those, which should be tackled more accurately --- eyes, mouth, etc)
* [ ] Sectored mask?
* [ ] Reparametrize batchnorm via gamma - 0.5, so it has zero mean
* [ ] Skip connections/batchnorm should make things worse, because they flatten the surface
* [x] Make scaling learnable
* [ ] Equal amount of both good and bad points in update
* [ ] What if we just optimize the differences between good and bad points? Maybe it's not equivalent to what we currently do?
* [ ] Try SGD
