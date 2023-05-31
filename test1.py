import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from torch.optim import AdamW
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReadabilityDataset(Dataset):
    def __init__(self, text, target=None):
        self.text = text
        self.target = target
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            return_attention_mask=True,
            truncation=True
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        if self.target is not None:
            target = self.target[idx]
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'target': torch.tensor(target, dtype=torch.float)
            }
        else:
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            }


class ReadabilityModel(torch.nn.Module):
    def __init__(self):
        super(ReadabilityModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = torch.nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooler_output = outputs.pooler_output
        output = self.linear(pooler_output)
        return output


def train_model(model, dataloader, optimizer, criterion, epochs):
    model.train()
    all_losses = []
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target = batch['target'].to(device)

            optimizer.zero_grad()
            output = model(input_ids, attention_mask)
            loss = criterion(output.view(-1), target)
            loss.backward()
            optimizer.step()
            all_losses.append(loss.item())
    return all_losses


def evaluate_model(model, dataloader):
    model.eval()
    preds = []
    actuals = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target = batch['target'].to(device)

            output = model(input_ids, attention_mask)
            preds.extend(output.view(-1).detach().cpu().numpy())
            actuals.extend(target.detach().cpu().numpy())
    return np.sqrt(mean_squared_error(actuals, preds)), preds, actuals


def predict_model(model, dataloader):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            output = model(input_ids, attention_mask)
            preds.extend(output.view(-1).detach().cpu().numpy())
    return preds


epochs = 1  # you can adjust the number of epochs

df = pd.read_csv("train.csv")

model_list = []  # 存储每个模型
model_rmse_list = []  # 存储每个模型的 RMSE

all_train_losses = []
kfold = KFold(n_splits=5, shuffle=True)
fold = 0
best_model = None
best_rmse = np.inf
for train_index, test_index in kfold.split(df):
    fold += 1
    print("Fold: ", fold)

    train_data = df.iloc[train_index]
    test_data = df.iloc[test_index]

    train_dataset = ReadabilityDataset(train_data["excerpt"].tolist(), train_data["target"].tolist())
    test_dataset = ReadabilityDataset(test_data["excerpt"].tolist(), test_data["target"].tolist())

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    model = ReadabilityModel().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    criterion = torch.nn.MSELoss()

    losses = train_model(model, train_dataloader, optimizer, criterion, epochs)
    all_train_losses.append(losses)  # 将当前fold的训练损失添加到列表中

    rmse, _, _ = evaluate_model(model, test_dataloader)
    print("RMSE: ", rmse)

    # 用新的模型替换当前的最佳模型，如果新模型的 RMSE 更低
    if rmse < best_rmse:
        best_rmse = rmse
        # 在存储新的最优模型前，删除旧的最优模型以释放 GPU 内存
        if best_model is not None:
            del best_model
            torch.cuda.empty_cache()  # 清空 GPU 缓存
        best_model = model
    else:
        # 如果当前模型不是最优模型，那么也可以直接删除以释放 GPU 内存
        del model
        torch.cuda.empty_cache()  # 清空 GPU 缓存

# 使用最好的模型对测试数据进行预测
submission_df = pd.read_csv("test.csv")
submission_dataset = ReadabilityDataset(submission_df["excerpt"].tolist())
submission_dataloader = DataLoader(submission_dataset, batch_size=4, shuffle=False)

submission_predictions = predict_model(best_model, submission_dataloader)

# 将预测结果存入 DataFrame
submission_df["target"] = submission_predictions

# 保存预测结果为 CSV 文件，以便提交到 Kaggle
submission_df.to_csv("submission1.csv", index=False)

# 在所有k-fold训练结束后，一次性画出所有的训练损失
fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # 修改 figsize 以增大图形窗口的大小
plt.subplots_adjust(hspace=0.5, wspace=0.4)  # 增加子图之间的间距
for i, losses in enumerate(all_train_losses):
    axs[i//3, i%3].plot(losses, label=f'Training Loss Fold {i+1}')  # 注意这里的修改，使得子图的索引能正确地映射到 2x3 的网格上
    axs[i//3, i%3].set_title(f'Training Loss Fold {i+1}', fontsize=10)  # 减小标题的字体大小
    axs[i//3, i%3].set_xlabel('Iteration', fontsize=8)  # 减小 x 轴标签的字体大小
    axs[i//3, i%3].set_ylabel('Loss', fontsize=8)  # 减小 y 轴标签的字体大小
    axs[i//3, i%3].legend(fontsize=8)  # 减小图例的字体大小
plt.tight_layout()
plt.show()

