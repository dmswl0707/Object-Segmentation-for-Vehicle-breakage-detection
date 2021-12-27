import torch
import torch.nn as nn
import time

class Semantic_Seg_Trainer(nn.Module):
    def __init__(self, model, opt="adam", num_class=2, lr=0.0001, has_scheduler=False, device="cpu", log_dir="./logs",
                 max_epoch=20):
        """
          Args:
            model: 사용할 model
            opt: optimizer
            lr: learning rate
            has_scheduler: learning rate scheduler 사용 여부
            device: 사용할 device (cpu/cuda)
        """
        super().__init__()

        self.max_epoch = max_epoch
        self.model = model
        self.loss = nn.CrossEntropyLoss()  # loss function 정의
        self.num_class = num_class

        self._get_optimizer(opt=opt.lower(), lr=lr)  # optimizer 정의
        self.has_scheduler = has_scheduler  # scheduler 사용여부
        if self.has_scheduler:
            self._get_scheduler()

        self.device = device  # 사용할 device

        self.log_dir = log_dir
        if not os.path.exists(log_dir): os.makedirs(log_dir)

    def _get_optimizer(self, opt, lr=0.001):
        """
          Args:
            opt: optimizer
            lr: learning rate
        """
        if opt == "sgd":
            self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=lr)
        elif opt == "adam":
            self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                              lr=lr)  # , weight_decay=Args["weight_decay"])
        else:
            raise ValueError(f"optimizer {opt} is not supproted")

    def _get_scheduler(self):
        # self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=5, gamma=0.5, verbose=True)
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=lambda epoch: 0.85**epoch)
        self.scheduler = CosineAnnealingWarmUpRestarts(optimizer=self.optimizer, T_0=10, T_mult=1,
                                                       eta_max=Args["eta_min"], T_up=2, gamma=0.3)

    def train(self, train_loader, valid_loader, max_epochs=20, disp_epoch=1, visualize=False):
        """
          네트워크를 학습시키는 함수
          Args:
            train_loader: 학습에 사용할 train dataloader
            valid_loader: validation에 사용할 dataloader
            max_epochs: 학습을 진행할 총 epoch 수
            disp_epochs: 학습 log를 display 할 epoch 주기
            visualize: 학습 진행 과정에서 결과 이미지를 visualize
        """
        print("===== Train Start =====")
        start_time = time.time()
        history = {"train_loss": [], "valid_loss": [], "train_miou": [], "valid_miou": []}

        for e in range(max_epochs):
            print(f"Start Train Epoch {e}")
            train_loss, train_miou = self._train_epoch(train_loader)
            print(f"Start Valid Epoch {e}")
            valid_loss, valid_miou = self._valid_epoch(valid_loader)

            history["train_loss"].append(train_loss)  # 현재 epoch에서 성능을 history dict에 저장
            history["valid_loss"].append(valid_loss)  #

            history["train_miou"].append(train_miou)  #
            history["valid_miou"].append(valid_miou)  #

            if self.has_scheduler:  # scheduler 사용할 경우 step size 조절
                self.scheduler.step()

            if e % disp_epoch == 0:  # disp_epoch 마다 결과값 출력
                print(
                    f"Epoch: {e}, train loss: {train_loss:>6f}, valid loss: {valid_loss:>6f}, train miou: {train_miou:>6f}, valid miou: {valid_miou:>6f}, time: {time.time() - start_time:>3f}")
                start_time = time.time()

            self.save_statedict(save_name=f"log_epoch_{e}")
            self.plot_history(history, save_name=f"{self.log_dir}/log_epoch_{e}.png")  # 그래프 출력

            #################################################################################################
            #                                                                                               #
            # TODO : 한 epoch 의 학습이 끝날때 마다 model 을 save 하는 코드를 작성해봅시다.                 #
            #        graph 저장 코드를 참고하여 저장되는 model 의 이름에 몇 epoch 인지 나타나게 해봅시다.   #
            #                                                                                               #
            #################################################################################################

    def _train_epoch(self, train_loader, disp_step=10):
        """
          model를 training set 한 epoch 만큼 학습시키는 함수
          Args:
            train_loader: 학습에 사용할 train dataloader
          Returns:
            training set 한 epoch의 평균 loss, 평균 accuracy
        """
        epoch_loss = 0

        miou = 0
        ious = np.zeros([2])

        self.model.train()  # self.model을 train 모드로 전환 --> nn.Module의 내장함수
        cnt = 0
        epoch_start_time = time.time()
        start_time = time.time()
        for (x, y) in train_loader:  # x: data, y:label
            cnt += 1

            x = x.to(self.device)
            label = y['masks'].to(self.device).type(torch.long)

            out = self.model(x)  # model이 예측한 output
            loss = self.loss(out['out'], label)

            self.optimizer.zero_grad()  # backwardpass를 통한 network parameter 업데이트
            loss.backward()  #
            self.optimizer.step()  #

            epoch_loss += loss.to("cpu").item()

            out_background = torch.argmin(out['out'].to("cpu"), dim=1).to(self.device)  # meanIoU 계산을 위한 데이터 변형
            out_target = torch.argmax(out['out'].to("cpu"), dim=1).to(self.device)  #

            ious[0] += self.batch_segmentation_iou(out_background,
                                                   torch.logical_not(label).type(torch.long))  # ious[0]:background IoU
            ious[1] += self.batch_segmentation_iou(out_target, label)  # ious[1]:파손 IoU

            if cnt % disp_step == 0:
                iou_back = ious[0] / (cnt * x.shape[0])
                iou_scratch = ious[1] / (cnt * x.shape[0])
                miou = (ious[0] / (cnt * x.shape[0]) + ious[1] / (cnt * x.shape[0])) / 2.

                print(
                    f"Iter: {cnt}/{len(train_loader)}, train epcoh loss: {epoch_loss / (cnt):>6f}, miou: {miou:>6f}, iou_back : {iou_back:>6f}, iou_scratch : {iou_scratch:>6f}, time: {time.time() - start_time:>3f}")
                start_time = time.time()

        epoch_loss /= len(train_loader)

        iou_back = ious[0] / (cnt * x.shape[0])
        iou_scratch = ious[1] / (cnt * x.shape[0])
        epoch_miou = (ious[0] / (cnt * x.shape[0]) + ious[1] / (cnt * x.shape[0])) / 2.
        print(
            f"Train loss: {epoch_loss:>6f}, miou: {epoch_miou:>6f}, iou_back : {iou_back:>6f}, iou_scratch : {iou_scratch:>6f}, time: {time.time() - epoch_start_time:>3f}")

        return epoch_loss, epoch_miou

    def _valid_epoch(self, valid_loader, disp_step=10):
        """
          현재 model의 성능을 validation set에서 측정하는 함수
          Args:
            valid_loader: 학습에 사용할 valid dataloader
          Returns:
            validation set 의 평균 loss, 평균 accuracy
        """
        epoch_loss = 0

        miou = 0
        ious = np.zeros([2])

        self.model.eval()  # self.model을 eval 모드로 전환 --> nn.Module의 내장함수
        cnt = 0
        epoch_start_time = time.time()
        start_time = time.time()
        with torch.no_grad():  # model에 loss의 gradient를 계산하지 않음
            for (x, y) in valid_loader:
                cnt += 1
                x = x.to(self.device)
                label = y['masks'].to(self.device).type(torch.long)

                out = self.model(x)
                loss = self.loss(out['out'], label)

                epoch_loss += loss.to("cpu").item()

                out_background = torch.argmin(out['out'].to("cpu"), dim=1).to(self.device)
                out_target = torch.argmax(out['out'].to("cpu"), dim=1).to(self.device)

                ious[0] += self.batch_segmentation_iou(out_background, torch.logical_not(label).type(torch.long))
                ious[1] += self.batch_segmentation_iou(out_target, label)

                if cnt % disp_step == 0:
                    iou_back = ious[0] / (cnt * x.shape[0])
                    iou_scratch = ious[1] / (cnt * x.shape[0])
                    miou = (ious[0] / (cnt * x.shape[0]) + ious[1] / (cnt * x.shape[0])) / 2.
                    print(
                        f"Iter: {cnt}/{len(valid_loader)}, valid epcoh loss: {epoch_loss / (cnt):>6f}, miou: {miou:>6f}, iou_back : {iou_back:>6f}, iou_scratch : {iou_scratch:>6f}, time: {time.time() - start_time:>3f}")
                    start_time = time.time()

        epoch_loss /= len(valid_loader)

        iou_back = ious[0] / (cnt * x.shape[0])
        iou_scratch = ious[1] / (cnt * x.shape[0])
        epoch_miou = (ious[0] / (cnt * x.shape[0]) + ious[1] / (cnt * x.shape[0])) / 2.
        print(
            f"Valid loss: {epoch_loss:>6f}, miou: {epoch_miou:>6f}, iou_back : {iou_back:>6f}, iou_scratch : {iou_scratch:>6f}, time: {time.time() - epoch_start_time:>3f}")

        return epoch_loss, epoch_miou

    def save_statedict(self, save_name=None):

        if not save_name == None:
            torch.save(seg_model.state_dict(), "/content/drive/MyDrive/Colab Notebooks/pth_path/" + save_name + ".pth")

    def plot_history(self, history, save_name=None):
        """
          history에 저장된 model의 성능을 graph로 plot
          Args:
            history: dictionary with keys {"train_loss","valid_loss",  }
                     각 item 들은 epoch 단위의 성능 history의 list
        """
        fig = plt.figure(figsize=(16, 8))

        ax = fig.add_subplot(1, 2, 1)
        ax.plot(history["train_loss"], color="red", label="train loss")
        ax.plot(history["valid_loss"], color="blue", label="valid loss")
        ax.title.set_text("Loss")
        ax.legend()

        ax = fig.add_subplot(1, 2, 2)
        ax.plot(history["train_miou"], color="red", label="train miou")
        ax.plot(history["valid_miou"], color="blue", label="valid miou")
        ax.title.set_text("miou")
        ax.legend()

        plt.show()

        if not save_name == None:  # graph 저장
            plt.savefig(save_name)

    def test(self, test_loader):
        """
          현재 model의 성능을 test set에서 측정하는 함수
          Args:
            test_loader: 학습에 사용할 test dataloader
          Returns:
            test set 의 평균 loss, 평균 accuracy
        """
        print("===== Test Start =====")
        start_time = time.time()
        epoch_loss = 0

        miou = 0
        ious = np.zeros([2])

        self.model.eval()  # self.model을 eval 모드로 전환 --> nn.Module의 내장함수
        cnt = 0
        epoch_start_time = time.time()
        start_time = time.time()
        with torch.no_grad():  # model에 loss의 gradient를 계산하지 않음
            for (x, y) in test_loader:
                cnt += 1
                x = x.to(self.device)
                label = y['masks'].to(self.device).type(torch.long)

                out = self.model(x)
                loss = self.loss(out['out'], label)

                epoch_loss += loss.to("cpu").item()

                out_background = torch.argmin(out['out'].to("cpu"), dim=1).to(self.device)
                out_target = torch.argmax(out['out'].to("cpu"), dim=1).to(self.device)

                ious[0] += self.batch_segmentation_iou(out_background, torch.logical_not(label).type(torch.long))
                ious[1] += self.batch_segmentation_iou(out_target, label)

                if cnt % 10 == 0:
                    iou_back = ious[0] / (cnt * x.shape[0])
                    iou_scratch = ious[1] / (cnt * x.shape[0])
                    miou = (ious[0] / (cnt * x.shape[0]) + ious[1] / (cnt * x.shape[0])) / 2.
                    print(
                        f"Iter: {cnt}/{len(valid_loader)}, test epcoh loss: {epoch_loss / (cnt):>6f}, miou: {miou:>6f}, iou_back : {iou_back:>6f}, iou_scratch : {iou_scratch:>6f}, time: {time.time() - start_time:>3f}")
                    start_time = time.time()

        epoch_loss /= len(test_loader)

        iou_back = ious[0] / (cnt * x.shape[0])
        iou_scratch = ious[1] / (cnt * x.shape[0])
        epoch_miou = (ious[0] / (cnt * x.shape[0]) + ious[1] / (cnt * x.shape[0])) / 2.

        print(
            f"Test loss: {epoch_loss:>6f}, miou: {epoch_miou:>6f}, iou_back : {iou_back:>6f}, iou_scratch : {iou_scratch:>6f}, time: {time.time() - epoch_start_time:>3f}")

    def batch_segmentation_iou(self, outputs, labels):
        """
            outputs, labels : (batch, h, w)
        """

        SMOOTH = 1e-6

        intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
        union = (outputs | labels).float().sum((1, 2))  # Will be zero if both are 0

        iou = (intersection + SMOOTH) / (union + SMOOTH)  # union = A+b - intersection

        #################################################################################################
        #                                                                                               #
        # TODO : 위 코드를 보고 IoU 를 계산하는 코드를 만들어봅시다.                                    #
        # hint : 나누기에서 0으로 나누면 error 가 발생하기 때문에 이를 피하기 위해 분자와 분모에        #
        #        아주 작은 수인 SMOOTH 를 더해줍시다                                                    #
        #        ex) x / y   --->   (x + SMOOTH) / (y + SMOOTH)                                         #
        #                                                                                               #
        #################################################################################################

        return torch.sum(iou).to("cpu").numpy()

