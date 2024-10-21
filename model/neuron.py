import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


class OurNeuralNetwork:
    def __init__(self):
        # Веса
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        self.w7 = np.random.normal()
        self.w8 = np.random.normal()
        self.w9 = np.random.normal()

        # Пороги
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def forward(self, x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2] + self.b1)
        h2 = sigmoid(self.w4 * x[0] + self.w5 * x[1] + self.w6 * x[2] + self.b2)
        o1 = sigmoid(self.w7 * h1 + self.w8 * h2 + self.b3)
        return o1

    def train(self, data, all_y_trues):
        learn_rate = 0.1
        epochs = 500

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2] + self.b1
                h1 = sigmoid(sum_h1)
                sum_h2 = self.w4 * x[0] + self.w5 * x[1] + self.w6 * x[2] + self.b2
                h2 = sigmoid(sum_h2)
                sum_o1 = self.w7 * h1 + self.w8 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                d_L_d_ypred = -2 * (y_true - y_pred)

                # Нейрон o1
                d_ypred_d_w7 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w8 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w7 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w8 * deriv_sigmoid(sum_o1)

                # Нейрон h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_w3 = x[2] * deriv_sigmoid(sum_h1)

                # Нейрон h2
                d_h2_d_w4 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w5 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_w6 = x[2] * deriv_sigmoid(sum_h2)

                # Обновляем веса и пороги
                # Нейрон 1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w3
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * deriv_sigmoid(sum_h1)

                # Нейрон 2
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w6
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * deriv_sigmoid(sum_h2)

                # Нейрон o1
                self.w7 -= learn_rate * d_L_d_ypred * d_ypred_d_w7
                self.w8 -= learn_rate * d_L_d_ypred * d_ypred_d_w8
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.forward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print(f"Epoch {epoch} loss: {loss:.3f}")

    def save_weights(self, filename):
        # Сохраняем веса и пороги в файл
        weights_and_biases = np.array([self.w1, self.w2, self.w3,
                                       self.w4, self.w5, self.w6,
                                       self.w7, self.w8, self.b1,
                                       self.b2, self.b3])
        np.savetxt(filename, weights_and_biases)

    def load_weights(self, filename):
        # Загружаем веса и пороги из файла
        data = np.loadtxt(filename)
        self.w1, self.w2, self.w3, self.w4, self.w5, self.w6, self.w7, self.w8, self.b1, self.b2, self.b3 = data
