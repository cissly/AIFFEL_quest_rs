# training.py
import os
import torch
import torch.optim as optim
import math
import torch.nn as nn
from config import MODEL_PATH, IMAGE_SHAPE
from data import create_dataloader
from models import StackedHourglassNetwork, SimpleBaselinePose
from tqdm import tqdm


class Trainer(object):
    def __init__(self,
                 model,
                 epochs,
                 global_batch_size,
                 initial_learning_rate):
        """
        - model: 학습시킬 PyTorch 모델(nn.Module)
        - epochs: 전체 학습 epoch 수
        - global_batch_size: 전체 배치 크기 (loss 계산 시 사용)
        - initial_learning_rate: 초기 학습률
        """
        self.model = model
        self.epochs = epochs
        self.global_batch_size = global_batch_size
        self.simple = True
        # MSE loss를 reduction='none'으로 사용 (가중치 적용을 위해)
        self.loss_object = nn.MSELoss(reduction='none')
        self.train_history = []
        self.val_history = []
        # Adam optimizer 초기화
        self.optimizer = optim.Adam(self.model.parameters(), lr=initial_learning_rate)

        # 학습률 스케줄링 관련 변수들
        self.current_learning_rate = initial_learning_rate
        self.last_val_loss = math.inf
        self.lowest_val_loss = math.inf
        self.patience_count = 0
        self.max_patience = 10

        # 최적 모델 체크포인트 저장
        self.best_model = None

        # 단일 GPU/멀티 GPU(DataParallel) 설정
        if torch.cuda.device_count() > 1:
            print(f"멀티 GPU 사용 (GPU 개수: {torch.cuda.device_count()})")
            self.model = nn.DataParallel(self.model)
        else:
            print("단일 GPU 혹은 CPU 사용")

    def lr_decay(self):
        """
        patience_count가 max_patience를 넘으면 학습률을 1/10으로 감소,
        그렇지 않고 val_loss가 그대로면 patience_count += 1,
        새 최저 val_loss를 달성하면 patience_count를 0으로.
        """
        if self.patience_count >= self.max_patience:
            self.current_learning_rate /= 10.0
            self.patience_count = 0
        elif self.last_val_loss == self.lowest_val_loss:
            self.patience_count = 0

        self.patience_count += 1

        # optimizer의 learning rate 갱신
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_learning_rate

    def lr_decay_step(self, epoch):
        """
        25, 50, 75 epoch에서 학습률을 1/10으로 감소시키는 스케줄링.
        """
        if epoch in [25, 50, 75]:
            self.current_learning_rate /= 10.0

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_learning_rate

    def compute_loss(self, labels, outputs):
        """
        여러 스택의 heatmap 출력(outputs)에 대해 MSE를 구하되,
        labels > 0인 위치에는 81의 추가 가중치를 적용.

        - Hourglass: outputs = [heatmap1, heatmap2, ...]
        - SimpleBaseline: outputs = heatmap (tensor)
        """
        loss = 0.0

        # 1) Hourglass (list of tensors)
        if isinstance(outputs, (list, tuple)):
            self.simple = False
            for output in outputs:
                weights = (labels > 0).float() * 81 + 1
                squared_error = (labels - output) ** 2
                weighted_error = squared_error * weights
                loss += weighted_error.mean() / self.global_batch_size

        # 2) SimpleBaseline (single tensor)
        elif torch.is_tensor(outputs):
            self.simple = True
            weights = (labels > 0).float() * 81 + 1
            squared_error = (labels - outputs) ** 2
            weighted_error = squared_error * weights
            loss += weighted_error.mean() / self.global_batch_size

        else:
            raise TypeError(f"Unexpected outputs type: {type(outputs)}")
        return loss

    def train_step(self, images, labels, device):
        self.model.train()
        images = images.to(device)
        labels = labels.to(device)

        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.compute_loss(labels, outputs)
        loss.backward()
        self.optimizer.step()

        self.train_history.append(loss.item())

        return loss.item()

    def val_step(self, images, labels, device):
        self.model.eval()
        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device)
            outputs = self.model(images)
            loss = self.compute_loss(labels, outputs)
        self.val_history.append(loss.item())
        return loss.item()

    def run(self, train_loader, val_loader, device):
        """
        - train_loader, val_loader: PyTorch DataLoader
        - device: torch.device('cuda' or 'cpu')
        """
        for epoch in range(1, self.epochs + 1):
            # 학습률 감소 로직
            self.lr_decay()
            print(f"\n===== Epoch {epoch} | lr={self.current_learning_rate:.6f} =====")

            # -------------------- Training --------------------
            total_train_loss = 0.0
            num_train_batches = 0

            train_pbar = tqdm(
                train_loader,
                desc=f"[Train] Epoch {epoch}",
                leave=False
            )
            for images, labels in train_pbar:
                batch_loss = self.train_step(images, labels, device)
                total_train_loss += batch_loss
                num_train_batches += 1
                avg_loss = total_train_loss / num_train_batches

                # 진행바에 현재/평균 loss 표시
                train_pbar.set_postfix({
                    "batch_loss": f"{batch_loss:.4f}",
                    "avg_loss": f"{avg_loss:.4f}"
                })

            train_loss = total_train_loss / max(1, num_train_batches)
            print(f"[Train] Epoch {epoch} loss: {train_loss:.4f}")

            # -------------------- Validation --------------------
            total_val_loss = 0.0
            num_val_batches = 0

            val_pbar = tqdm(
                val_loader,
                desc=f"[Val]   Epoch {epoch}",
                leave=False
            )
            for images, labels in val_pbar:
                batch_loss = self.val_step(images, labels, device)
                # NaN이면 카운트에서 제외
                if not math.isnan(batch_loss):
                    total_val_loss += batch_loss
                    num_val_batches += 1

                    avg_val_loss = total_val_loss / num_val_batches
                    val_pbar.set_postfix({
                        "batch_loss": f"{batch_loss:.4f}",
                        "avg_loss": f"{avg_val_loss:.4f}"
                    })

            if num_val_batches > 0:
                val_loss = total_val_loss / num_val_batches
            else:
                val_loss = float('nan')

            print(f"[Val]   Epoch {epoch} loss: {val_loss:.4f}")

            # -------------------- Checkpoint --------------------
            if val_loss < self.lowest_val_loss:
                self.save_model(epoch, val_loss)
                self.lowest_val_loss = val_loss
            self.last_val_loss = val_loss

        return self.best_model, self.train_history, self.val_history

    def save_model(self, epoch, loss):
        if self.simple:
            model_name = os.path.join(MODEL_PATH, f'simple-model-epoch-{epoch}-loss-{loss:.4f}.pt')
        else:
            model_name = os.path.join(MODEL_PATH, f'Hourglass-model-epoch-{epoch}-loss-{loss:.4f}.pt')
        torch.save(self.model.state_dict(), model_name)
        self.best_model = model_name
        print(f"Model {model_name} saved.")


def train(epochs,
          learning_rate,
          num_heatmap,
          batch_size,
          train_annotation_file,
          val_annotation_file,
          image_dir,
          simple: bool = False):
    """
    - epochs: 전체 학습 epoch 수
    - learning_rate: 초기 학습률
    - num_heatmap: 생성할 heatmap 개수 (num_joints)
    - batch_size: 배치 크기
    - train_annotation_file: train.json 파일 경로
    - val_annotation_file: validation.json 파일 경로
    - image_dir: 이미지 파일들이 저장된 디렉토리 경로
    - simple: True이면 SimpleBaselinePose, False이면 StackedHourglassNetwork 사용
    """
    global_batch_size = batch_size

    train_loader = create_dataloader(
        train_annotation_file, image_dir, batch_size, num_heatmap, is_train=True
    )
    val_loader = create_dataloader(
        val_annotation_file, image_dir, batch_size, num_heatmap, is_train=False
    )

    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    # ------------------ 모델 선택 부분 ------------------
    if simple:
        print("SimpleBaselinePose 모델로 학습을 진행합니다.")
        model = SimpleBaselinePose(
            num_joints=num_heatmap,
            backbone="resnet50",
            pretrained=False,
        )
    else:
        print("StackedHourglassNetwork 모델로 학습을 진행합니다.")
        model = StackedHourglassNetwork(
            IMAGE_SHAPE,
            num_stack=4,
            num_residual=1,
            num_heatmap=num_heatmap
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    trainer = Trainer(
        model,
        epochs,
        global_batch_size,
        initial_learning_rate=learning_rate
    )

    print("Start training...")
    return trainer.run(train_loader, val_loader, device)