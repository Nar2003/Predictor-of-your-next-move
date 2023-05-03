import numpy as np

data = np.zeros((20,1))

import keras
from keras.layers import Dense,Input

model = keras.Sequential([
    Input((19)),
    Dense(8,activation= 'relu'),
    Dense(1,activation= 'sigmoid')
])
model.summary()
model.compile(
    loss = 'mean_squared_error',
    optimizer='adam'
)


while True:
    # print (data)
    model.fit(
        data[:-1].reshape(1,19),
        data[-1].reshape(1,1),
        verbose=False
    )
    data = data[1:]
    print(f'\n think your next move is {round(model.predict(data.reshape(1,19),verbose=False)[0][0])}')
    data = np.append(data,np.array([int(input('\n0 or 1 \n\n'))]).reshape(1,1))

