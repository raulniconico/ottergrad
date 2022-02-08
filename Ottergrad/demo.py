# if node.gettype() == np.dot or node.gettype() == __matmul__:
                #     node.getleft().grad = np.dot(node.getgrad(), node.getright().getdata().T)
                #     node.getright().grad = np.dot(node.getleft().getdata().T,  node.getgrad())
                #
                # # elif node.gettype() == Sigmoid:
                # #     node.getleft().grad = np.multiply(node.getgrad(), np.multiply(Sigmoid(node.getgrad()),
                # #                                                                   1 - Sigmoid(node.getgrad())))
                # # elif node.gettype() == ReLU:
                # #     grad = copy.deepcopy(node.getgrad())
                # #     grad[grad <= 0] = 0
                # #     node.getleft().grad = grad
                #
                # elif node.gettype() == np.tanh:
                #     node.getleft().grad = np.multiply(
                #         (np.ones(node.getgrad().shape) - np.tanh(node.getgrad()) ** 2),
                #         node.getgrad())
                #
                #
                # elif node.gettype() == ndarray.__add__ or node.gettype() == ndarray.__radd__:
                #     if node.getleft().getisconst():
                #         node.getleft().grad = np.sum(node.getgrad(), axis=0)
                #     else:
                #         node.getleft().grad = node.getgrad()
                #     if node.getright().getisconst():
                #         node.getright().grad = np.sum(node.getgrad(), axis=0)
                #     else:
                #         node.getright().grad = node.getgrad()
                #
                # elif node.gettype() == ndarray.__truediv__:
                #     node.getleft().grad = np.dot(np.divide(1, node.getright().getdata()), node.getgrad())
                #     node.getright().grad = np.dot(node.getleft().getdata(), -node.getright().getdata()**-2)
                #
                # elif node.gettype() == ndarray.__rtruediv__:
                #     node.getleft().grad = np.dot(node.getleft().getdata(), -node.getright().getdata()**-2)
                #     node.getright().grad = np.dot(np.divide(1, node.getright().getdata()), node.getgrad())
                #
                # elif node.gettype() == ndarray.__sub__:
                #     node.getleft().grad = node.getgrad()
                #     node.getright().grad = -node.getgrad()
                #
                # elif node.gettype() == ndarray.__rsub__:
                #     node.getleft().grad = -node.getgrad()
                #     node.getright().grad = node.getgrad()
                #
                # elif node.gettype() == ndarray.__neg__:
                #     node.getleft().grad = -node.getgrad()
                #
                # elif node.gettype() == "*scalar":
                #     node.getleft().grad = node.getright().getdata() * node.getgrad()
                #     node.getright().grad = np.sum(node.getleft().getdata().T, axis=0) @ node.getgrad()
                #
                # elif node.gettype() == "normalization":
                #     # cache = [x,  sigma_beta^2, mu_beta]
                #
                #     # dx = 1/N / std * (N * dx_norm -
                #     #       dx_norm.sum(axis=0) -
                #     #       x_norm * (dx_norm * x_norm).sum(axis=0))
                #
                #     x = node.cache[0]
                #     sigma2 = node.cache[1]
                #     mu = node.cache[2]
                #
                #     dl_dx_hat = node.getgrad()
                #     dl_dsigma2 = np.sum(dl_dx_hat, axis=0) * (x - mu) * -0.5 * (sigma2 + epsilon) ** -3 / 2
                #     dl_dmu = np.sum(dl_dx_hat, axis=0) * -1 / np.sqrt(sigma2 + epsilon) + dl_dsigma2 * np.sum(
                #         -2 * (x - mu),
                #         axis=0) / \
                #              x.shape[0]
                #     dl_dx = dl_dx_hat * 1 / np.sqrt(sigma2 + epsilon) + dl_dsigma2 * 2 * (x - mu) / x.shape[
                #         0] + dl_dmu / \
                #             x.shape[0]
                #     node.getleft().grad = dl_dx
                #
                # elif node.gettype() == np.exp:
                #     node.getleft().grad = node.getdata()


# @checktensor
# def dot(x, y):
#     tensor = ag.Tensor()
#     tensor.type = np.dot
#     tensor.left = x
#     tensor.right = y
#     tensor.kwargs = None
#     return tensor

# @checktensor
# def sum(x, **kwargs):
#     if type(x) is ag.Tensor:
#         tensor = ag.Tensor()
#         tensor.left = x
#         tensor.type = np.sum
#         tensor.kwargs = kwargs
#     else:
#         tensor = ag.Tensor(np.sum(x, kwargs))
#         tensor.type = np.sum
#         tensor.kwargs = kwargs
#     return tensor

# def ones(shape, dtype=getdtype()):
#     tensor = ag.Tensor(np.ones(shape))
#     tensor.type = np.ones
#     tensor.isgrad = False
#     tensor.kwargs = dtype
#     return tensor

# def shape(x):
#     if type(x) is ag.Tensor:
#         return np.shape(x.getdata())
#     else:
#         return np.shape(x)
#
# @checktensor
# def exp(x, **kwargs):
#     tensor = Tensor()
#
#     tensor.left = x
#     tensor.type = np.exp
#     tensor.setkwargs(kwargs)
#     return tensor



# @checktensor
# def maximum(x, y, **kwargs):
#     tensor = Tensor()
#
#     tensor.left = x
#     tensor.right = y
#     tensor.kwargs = kwargs
#     tensor.type = np.maximum
#     return tensor






# def forwardpropagation(self):
#     def forward(node):
#
#         if node.getargs() is not None:
#             try:
#                 args = []
#                 for arg in node.getargs():
#                     if type(arg) is Tensor:
#                         forward(arg)
#                         args.append(arg.getdata())
#                     else:
#                         args.append(arg)
#                 node.args = args
#
#             except:
#
#                 if type(node.getargs()) is Tensor:
#                     forward(node.getargs())
#                     node.args = node.getargs().getdata()
#
#         if node.getkwargs() is not None:
#             try:
#                 kwargs = []
#                 for kwarg in node.getkwargs():
#                     if type(kwarg) is Tensor:
#                         forward(kwarg)
#                         kwargs.append(kwarg.getdata())
#                     else:
#                         kwargs.append(kwarg)
#                 node.kwargs = kwargs
#
#             except:
#                 if type(node.getkwargs()) is Tensor:
#                     forward(node.getkwargs())
#                     node.kwargs = node.getkwargs().getdata()
#
#         if type(node.getdata()) is Tensor:
#             forward(node.getdata())
#             node.data = node.getdata().getdata()
#
#         if node.getleft() is not None:
#             if node.getleft().getdata() is None:
#                 forward(node.getleft())
#
#             if node.getright() is not None:
#                 if node.getright().getdata() is None:
#                     forward(node.getright())
#
#                 if node.getargs() is not None and node.getkwargs() is not None:
#
#                     node.data = node.gettype()(node.getleft().getdata(), node.getright().getdata(), *node.getargs(),
#                                                **node.getkwargs())
#                 elif node.getargs() is None and node.getkwargs() is not None:
#                     node.data = node.gettype()(node.getleft().getdata(), node.getright().getdata(),
#                                                **node.getkwargs())
#                 elif node.getargs() is not None and node.getkwargs() is None:
#                     node.data = node.gettype()(node.getleft().getdata(), node.getright().getdata(),
#                                                *node.getargs())
#                 elif node.getargs() is None and node.getkwargs() is None:
#                     node.data = node.gettype()(node.getleft().getdata(), node.getright().getdata())
#             else:
#
#                 if node.getargs() is not None and node.getkwargs() is not None:
#
#                     node.data = node.gettype()(node.getleft().getdata(), *node.getargs(), **node.getkwargs())
#                 elif node.getargs() is None and node.getkwargs() is not None:
#                     node.data = node.gettype()(node.getleft().getdata(), **node.getkwargs())
#                 elif node.getargs() is not None and node.getkwargs() is None:
#                     node.data = node.gettype()(node.getleft().getdata(),
#                                                *node.getargs())
#                 elif node.getargs() is None and node.getkwargs() is None:
#                     node.data = node.gettype()(node.getleft().getdata())
#                 # node.data = node.gettype()(node.getleft().getdata(), *node.getargs(), **node.getkwargs())
#
#         elif node.getright() is None:
#             if node.getargs() is not None and node.getkwargs() is not None:
#                 node.data = node.gettype()(node.getdata(), *node.getargs(), **node.getkwargs())
#             elif node.getargs() is None and node.getkwargs() is not None:
#                 node.data = node.gettype()(node.getdata(), **node.getkwargs())
#             elif node.getargs() is not None and node.getkwargs() is None:
#                 node.data = node.gettype()(node.getdata(),
#                                            *node.getargs())
#             elif node.getargs() is None and node.getkwargs() is None:
#                 node.data = node.gettype()(node.getdata())
#
#         return
#
#     forward(self.getroot())
#     return self.getroot()