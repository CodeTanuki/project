import os
import tensorflow as tf
import matplotlib.pyplot as plt
# train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
# train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),origin=train_dataset_url)
# print("데이터셋이 복사된 위치: {}".format(train_dataset_fp))
# !head -n5 {train_dataset_fp}   # head n5 명령을 통해 처음 5개 항목을 확인, 4개의 특성(Feature)과 3개의 레이블(Label)
train_dataset_fp = 'D:/대학교/2학년 2학기/K-mooc/python/Iris Classification/iris_training.csv'
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
feature_names = column_names[:-1]      # [:-1]은 처음부터 끝까지
label_name = column_names[-1]          # [-1]은 끝의 것, 즉 'species'를 가지고 옴
# print("특성: {}".format(feature_names))
# print("레이블: {}".format(label_name))
class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
batch_size = 64
# 4개의 Feature값들에 대한 배치사이즈 만큼의 데이터 배열을 확인 할 수 있음
train_dataset = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)
features, labels = next(iter(train_dataset))
print(features)
# batch_size만큼의 데이터의 산포도를 그려 특성을 확인
plt.scatter(features['petal_length'],
            features['sepal_length'],
            c=labels,
            cmap='viridis')
plt.xlabel("Petal length")
plt.ylabel("Sepal length")
plt.show()
# 특성들을 단일 배열로 묶는다
def pack_features_vector(features, labels):
  features = tf.stack(list(features.values()), axis=1)
  return features, labels
# (feature, label)쌍의 특성을 훈련 데이터 세트에 쌓기위해 tf.data.Dataset.map 메서드를 사용
train_dataset = train_dataset.map(pack_features_vector)
# 데이터셋의 특성 요소의 형태가 (batch_size, num_features)인 배열로 바뀜, 즉 4개의 feature씩 묶여, 입력데이터 셋으로 바뀜
# features, labels = next(iter(train_dataset))
# print(features[:5])

model = tf.keras.Sequential([          # Neural Net을 구성, MINIST, CIFAR쪽 코드와 비교해 볼 것
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # 입력의 형태가 필요
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(3)
])
# predictions = model(features)        # 일단 예측은 가능하나 훈련을 통한 최적화가 이루어 지지 않아 거의 틀린 닶을 내놈
# predictions[:5]
# tf.nn.softmax(predictions[:5])
# print("  예측: {}".format(tf.argmax(predictions, axis=1)))
# print("레이블: {}".format(labels))   # 일단 예측은 가능하나 훈련을 통한 최적화가 이루어 지지 않아 거의 틀린 닶을 내놈

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
def loss(model, x, y):              # 손실함수 정의
  y_ = model(x)
  return loss_object(y_true=y, y_pred=y_)
def grad(model, inputs, targets):   # 최적구배법을 이용한 최적화 함수 정의
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)  # 최적화를 위한 Optimizer설정

# 도식화를 위해 결과를 저장
train_loss_results = []
train_accuracy_results = []
num_epochs = 201
for epoch in range(num_epochs):
  epoch_loss_avg = tf.keras.metrics.Mean()
  epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
  for x, y in train_dataset:
    loss_value, grads = grad(model, x, y)   # 모델을 최적화
    optimizer.apply_gradients(zip(grads, model.trainable_variables))  
    epoch_loss_avg(loss_value)              # 진행 상황을 추적. 현재 배치 손실을 추가. 
    epoch_accuracy(y, model(x))             # 예측된 레이블과 실제 레이블 비교합니다.
  # epoch 종료
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())
  if epoch % 50 == 0:
    print("에포크 {:03d}: 손실: {:.3f}, 정확도: {:.3%}".format(epoch,epoch_loss_avg.result(),epoch_accuracy.result()))

# 손실함수 시각화
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Error and Accuracy with Epoch processing')
axes[0].set_ylabel("Error", fontsize=14)
axes[0].plot(train_loss_results)
axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()

# 테스트 데이터 설정
test_fp = '/python my project/iris_test.csv'
test_dataset = tf.data.experimental.make_csv_dataset(
    test_fp,
    batch_size,
    column_names=column_names,
    label_name='species',
    num_epochs=1,
    shuffle=False)
test_dataset = test_dataset.map(pack_features_vector)

# 테스트 모델 평가
test_accuracy = tf.keras.metrics.Accuracy()

for (x, y) in test_dataset:
  logits = model(x)
  prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)
print("테스트 세트 정확도: {:.3%}".format(test_accuracy.result()))

# 레이블 되지 않은 임의의 샘플을 제공하여 평가해보기
predict_dataset = tf.convert_to_tensor([
    [5.1, 3.3, 1.7, 0.5,],
    [5.9, 3.0, 4.2, 1.5,],
    [6.9, 3.1, 5.4, 2.1]
])

predictions = model(predict_dataset)

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  p = tf.nn.softmax(logits)[class_idx]
  name = class_names[class_idx]
  print("샘플 {} 예측: {} ({:4.1f}%)".format(i, name, 100*p))