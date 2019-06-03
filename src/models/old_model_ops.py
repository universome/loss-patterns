# class ModelWithWeightsSamplingOnCircle(ModelOperation):
#     def __init__(self, theta, r, center, param_sizes):
#         self.theta = theta
#         self.r = r
#         self.center = center
#         self.param_sizes = param_sizes

#     def __call__(self, X):
#         params = sample_params(self.theta, self.r, self.center, self.param_sizes)
#         model = SuperModelOperation(params)

#         return model(X)
#       def sample(self):
#         z = orthogonalize(self.theta, self.r, adjust_len=True)
#         #w = sample_on_circle(self.center, z, self.r)
#         #w = sample_on_square(self.center, z, self.r)
#         w = self.center + z
#         params = weight_to_param(w, self.param_sizes)

#         return params

#     def to(self, *args, **kwargs):
#         self.theta = self.theta.to(*args, **kwargs)
#         self.r = self.r.to(*args, **kwargs)
#         self.center = self.center.to(*args, **kwargs)

#         return self


# class CircleModel(ModelOperation):
#     def __init__(self, r, center):
#         self.theta = nn.Parameter(weight_vector(SuperModel().parameters())) # Good init
#         self.center = nn.Parameter(center)
#         self.r = r
#         self.param_sizes = param_sizes(SuperModel().parameters())

#     def __call__(self, x):
#         if self.training:
#             z = self.theta - self.center
#             w = sample_on_circle(self.center, z, self.r)
#         else:
#             w = self.theta

#         params = weight_to_param(w, self.param_sizes)
#         model = SuperModelOperation(params, training=self.training)

#         return model(x)

#     def compute_reg(self):
#         ort_reg = torch.dot(self.theta, self.r).pow(2)
#         norm_reg = (torch.norm(self.theta) - torch.norm(self.r)).pow(2)

#         return norm_reg, ort_reg

#     def to(self, *args, **kwargs):
#         self.theta = nn.Parameter(self.theta.to(*args, **kwargs))
#         self.r = self.r.to(*args, **kwargs)
#         self.center = self.center.to(*args, **kwargs)

#         return self


# class TwoCirclesModel(ModelOperation):
#     "Our theta is the head of the first circle"
#     def __init__(self):
#         self.w_a = weight_vector(SuperModel().parameters())
#         self.w_b = weight_vector(SuperModel().parameters())
#         self.theta = nn.Parameter(weight_vector(SuperModel().parameters())) # Good init
#         self.param_sizes = param_sizes(SuperModel().parameters())

#     @property
#     def r(self):
#         return (self.w_b - self.w_a) / 2

#     @property
#     def center(self):
#         return self.w_a + self.r

#     def __call__(self, x):
#         if self.training:
#             w = self.sample()
#         else:
#             w = self.center

#         params = weight_to_param(w, self.param_sizes)
#         model = SuperModelOperation(params, training=self.training)

#         return model(x)

#     def sample(self):
#         if np.random.rand() > 0.5:
#             center = self.first_circle_center()
#         else:
#             center = self.second_circle_center()

#         return sample_on_circle(center, self.theta, 0.5 * self.r)

#     def first_circle_center(self):
#         return self.center - 0.5 * self.r

#     def second_circle_center(self):
#         return self.center + 0.5 * self.r

#     def compute_reg(self):
#         ort_reg = torch.dot(self.theta, self.r).pow(2)
#         norm_reg = (torch.norm(self.theta) - 0.5 * torch.norm(self.r)).pow(2)

#         return norm_reg, ort_reg

#     def to(self, *args, **kwargs):
#         self.w_a = nn.Parameter(self.w_a.to(*args, **kwargs))
#         self.w_b = nn.Parameter(self.w_b.to(*args, **kwargs))
#         self.theta = nn.Parameter(self.theta.to(*args, **kwargs))

#         return self

#     def run_from_weights(self, w, x):
#         params = weight_to_param(w, self.param_sizes)
#         model = SuperModelOperation(params, training=self.training)

#         return model(x)

#     def sample_on_outer_circle(self):
#         return sample_on_circle(self.center, self.theta * 4, self.r * 2)


# class LineModel(ModelOperation):
#     def __init__(self):
#         self.w_a = nn.Parameter(weight_vector(SuperModel().parameters()))
#         self.w_b = nn.Parameter(weight_vector(SuperModel().parameters()))
#         self.param_sizes = param_sizes(SuperModel().parameters())

#     def get_distance(self):
#         return (self.w_a - self.w_b).norm()

#     def __call__(self, x):
#         if self.training:
#             alpha = np.random.rand()
#             w = (1 - alpha) * self.w_a + alpha * self.w_b
#         else:
#             w = 0.5 * self.w_a + 0.5 * self.w_b

#         params = weight_to_param(w, self.param_sizes)
#         model = SuperModelOperation(params, training=self.training)

#         return model(x)

#     def to(self, *args, **kwargs):
#         self.w_a = nn.Parameter(self.w_a.to(*args, **kwargs))
#         self.w_b = nn.Parameter(self.w_b.to(*args, **kwargs))

#         return self


# class ElbowModel(ModelOperation):
#     def __init__(self):
#         self.w_a = nn.Parameter(weight_vector(SuperModel().parameters()))
#         self.w_b = nn.Parameter(weight_vector(SuperModel().parameters()))
#         self.w_c = nn.Parameter(weight_vector(SuperModel().parameters()))
#         self.param_sizes = param_sizes(SuperModel().parameters())

#     def sample(self):
#         alpha = np.random.random()
#         beta = np.random.random()

#         # Randomly choosing a link
#         min_to_use = self.w_a if beta > 0.5 else self.w_b

#         # Randomly choosing a point on a link
#         w = min_to_use * (1 - alpha) + self.w_c * alpha

#         return w

#     def run_from_weights(self, w, x):
#         params = weight_to_param(w, self.param_sizes)
#         model = SuperModelOperation(params, training=self.training)

#         return model(x)

#     def __call__(self, x):
#         if self.training:
#             w = self.sample()
#         else:
#             w = self.w_c

#         return self.run_from_weights(w, x)

#     def to(self, *args, **kwargs):
#         self.w_a = nn.Parameter(self.w_a.to(*args, **kwargs))
#         self.w_b = nn.Parameter(self.w_b.to(*args, **kwargs))
#         self.w_c = nn.Parameter(self.w_c.to(*args, **kwargs))

#         return self


# class ElbowTinyModel(ElbowModel):
#     def __init__(self, num_hidden:int):
#         self.w_a = nn.Parameter(weight_vector(TinySuperModel(num_hidden).parameters()))
#         self.w_b = nn.Parameter(weight_vector(TinySuperModel(num_hidden).parameters()))
#         self.w_c = nn.Parameter(weight_vector(TinySuperModel(num_hidden).parameters()))
#         self.param_sizes = param_sizes(TinySuperModel(num_hidden).parameters())

#     def run_from_weights(self, w, x):
#         params = weight_to_param(w, self.param_sizes)
#         model = TinySuperModelOperation(params, training=self.training)

#         return model(x)
