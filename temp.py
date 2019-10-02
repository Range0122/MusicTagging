from models import *

input_shape = (96, 1366, 1)
output_class = 50
model = TestRes(input_shape, output_class)
model.summary()
