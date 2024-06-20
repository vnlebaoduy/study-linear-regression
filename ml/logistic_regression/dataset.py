from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

X_digits, y_digits = load_digits(n_class=10, return_X_y=True)
# Định nghĩa số lượng hình ảnh muốn hiển thị
num_images = 10
num_cols = 5
num_rows = 2
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 4))

for i in range(num_images):
    row = i // num_cols
    col = i % num_cols
    axes[row, col].imshow(X_digits[i].reshape(8, 8), cmap='gray')
    axes[row, col].set_title(y_digits[i])
    axes[row, col].axis('off')

# Ẩn các subplot dư thừa nếu có
for i in range(num_images, num_rows * num_cols):
    row = i // num_cols
    col = i % num_cols
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()
