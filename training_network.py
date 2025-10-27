import numpy as np


#     write to training_results.txt rather than printing for end results to turn in 


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    return x * (1 - x)


def notX2():
    X = np.array([
        [1, 0],  # bias always 1
        [1, 1],  # input can be 0 or 1 from X2
    ])
    #X1 = X[:, 0]
    #X2 = X[:, 1]

    y = np.array([[1], [0]])  # target output is 1 or 0 if input is 0 or 1
    w = np.array([[1.0], [1.0]])  # weights where w[0] is the bias weight and w[1] is the input weight
    learning_rate = 1.0

    for epoch in range(2000):  # go through 2000 passes
        total_dE_dw = np.zeros_like(w)
        total_error = 0  # reset error for each pass

        for i in range(len(X)):
            x = X[i].reshape(1, 2)  # shape (1, 2)
            target = y[i]  # scalar target value

            # ---- Forward pass ----
            net = np.dot(x, w)  # weighted input (bias + input * weight)
            out = sigmoid(net)  # neuron output between 0–1

            # ---- Error ----
            error = out - target
            sq_err = error ** 2
            total_error += sq_err[0][0]  # add to total error

            # ---- Derivative wrt each weight ----
            dE_dw = 2 * error * sigmoid_deriv(out) * x.T

            # accumulate gradients across both examples
            total_dE_dw += dE_dw

            # ---- After both inputs, update weights ----
        w -= learning_rate * total_dE_dw

        if epoch % 100 == 99:
            print(f"XOR Epoch {epoch + 1:2d}: weights = {w.flatten()}, total_error = {total_error:.6f}")

    print("\nFinal evaluation for XOR:")
    for i in range(len(X)):
        net = np.dot(X[i], w)
        out = sigmoid(net)
        print(f"Input X2={X[i][1]} → Output={out[0]:.4f}")


def main():
    notX2()


if __name__ == "__main__":
    main()