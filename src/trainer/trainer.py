import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, model, device, hypergraphs):
        self.model = model.to(device)
        self.device = device
        self.hypergraphs = hypergraphs

    def train(self, train_loader, val_loader, test_loader, evaluator, epochs, lr, save_path):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0

            for batch in tqdm(train_loader):
                items, cats, item_targets, cat_targets, masks, times = batch

                items = items.to(self.device)
                cats = cats.to(self.device)
                item_targets = item_targets.to(self.device)
                cat_targets = cat_targets.to(self.device)

                optimizer.zero_grad()

                item_pred, cat_pred = self.model(
                    items, cats, self.hypergraphs, times, masks
                )

                loss = self.model.loss(item_pred, cat_pred, item_targets, cat_targets)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

            self.evaluate(val_loader, evaluator, "Validation")

        torch.save(self.model.state_dict(), save_path)
        print("Model saved!")

        self.evaluate(test_loader, evaluator, "Test")

    def evaluate(self, loader, evaluator, mode):
        results = evaluator.evaluate(self.model, loader, self.device, self.hypergraphs)
        print(f"[{mode}] {results}")