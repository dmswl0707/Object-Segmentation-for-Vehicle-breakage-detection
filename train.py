from trainer import *
from args import *

device = "cuda"
trainer = Semantic_Seg_Trainer(model=seg_model, opt="Adam", lr=Args["lr"], has_scheduler=True, device=device).to(device)
start_time = time.time()
#trainer.train(train_loader, valid_loader, max_epochs=Args["max_epochs"], disp_epoch=1)
print(f"Training time : {time.time()-start_time:>3f}")
#################################################################################################
#                                                                                               #
# TODO : trainer 를 정의해봅시다.                                                               #
#                                                                                               #
#################################################################################################
