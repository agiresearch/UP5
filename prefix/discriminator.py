import torch
import torch.nn as nn
import os


class BinaryDiscriminator(nn.Module):
    @staticmethod
    def parse_disc_args(parser):
        parser.add_argument(
            "--neg_slope", type=float, default=0.2, help="negative slope for leakyReLU."
        )
        return parser

    def __init__(
        self, args,
    ):
        super().__init__()
        self.args = args

        self.criterion = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.args.d_model = 512

        # For Ml-1M
        self.network = nn.Sequential(
            nn.Linear(self.args.d_model, int(self.args.d_model * 2), bias=True),
            nn.BatchNorm1d(num_features=self.args.d_model * 2),
            nn.LeakyReLU(0.2),
            # nn.Dropout(p=0.3),
            nn.Linear(
                int(self.args.d_model * 2), int(self.args.d_model * 4), bias=True
            ),
            nn.LeakyReLU(0.2),
            # nn.Dropout(p=0.3),
            nn.Linear(
                int(self.args.d_model * 4), int(self.args.d_model * 4), bias=True
            ),
            nn.LeakyReLU(0.2),
            # nn.Dropout(p=0.3),
            nn.Linear(
                int(self.args.d_model * 4), int(self.args.d_model * 2), bias=True
            ),
            nn.BatchNorm1d(num_features=self.args.d_model * 2),
            nn.LeakyReLU(0.2),
            # nn.Dropout(p=0.3),
            nn.Linear(
                int(self.args.d_model * 2), int(self.args.d_model * 2), bias=True
            ),
            nn.BatchNorm1d(num_features=self.args.d_model * 2),
            nn.LeakyReLU(0.2),
            # nn.Dropout(p=0.3),
            nn.Linear(int(self.args.d_model * 2), int(self.args.d_model), bias=True),
            nn.LeakyReLU(0.2),
            # nn.Dropout(p=0.3),
            nn.Linear(int(self.args.d_model), int(self.args.d_model), bias=True),
            nn.LeakyReLU(0.2),
            # nn.Dropout(p=0.3),
            nn.Linear(int(self.args.d_model), int(self.args.d_model / 2), bias=True),
            nn.LeakyReLU(0.2),
            # nn.Dropout(p=0.3),
            nn.Linear(int(self.args.d_model / 2), 1, bias=True),
        )

    @staticmethod
    def init_weights(m):
        """
        initialize nn weights，called in main.py
        :param m: parameter or the nn
        :return:
        """
        if type(m) == torch.nn.Linear:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, mean=0.0, std=0)
        elif type(m) == torch.nn.Embedding:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)

    def forward(self, embeddings, labels):
        output = self.predict(embeddings)["output"]
        if torch.cuda.device_count() > 0:
            labels = labels.cpu().type(torch.FloatTensor).cuda()
        else:
            labels = labels.type(torch.FloatTensor)
        if output.squeeze().dim() == 0:
            loss = self.criterion(output.squeeze().unsqueeze(0), labels)
        else:
            loss = self.criterion(output.squeeze(), labels)
        return loss

    def predict(self, embeddings):
        scores = self.network(embeddings)
        output = self.sigmoid(scores).squeeze()

        # prediction = output.max(1, keepdim=True)[1]     # get the index of the max
        if torch.cuda.device_count() > 0:
            threshold = torch.tensor(0.5).cuda()
        else:
            threshold = torch.tensor(0.5)
        prediction = (output > threshold).float() * 1

        result_dict = {"output": output, "prediction": prediction}
        return result_dict

    """
    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        dir_path = os.path.dirname(model_path)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        torch.save(self.state_dict(), model_path)
        # logging.info('Save ' + self.name + ' discriminator model to ' + model_path)

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        self.load_state_dict(torch.load(model_path))
        self.eval()
        logging.info("Load " + self.name + " discriminator model from " + model_path)
    """


class MultiDiscriminator(nn.Module):
    @staticmethod
    def parse_disc_args(parser):
        parser.add_argument(
            "--neg_slope", type=float, default=0.2, help="negative slope for leakyReLU."
        )
        return parser

    def __init__(self, args, labels=None):
        super().__init__()
        self.args = args
        if labels is not None:
            self.labels = labels
        else:
            if args.task == "insurance":
                if "marital" in self.args.feature:
                    self.labels = 3
                elif "occupation" in self.args.feature:
                    self.labels = 3
                else:
                    assert "age" in self.args.feature
                    self.labels = 5
            else:
                if "age" in args.feature:
                    self.labels = 7
                elif "occupation" in self.args.feature:
                    self.labels = 15

        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.args.d_model = 512

        self.network = nn.Sequential(
            nn.Linear(self.args.d_model, int(self.args.d_model * 2), bias=True),
            nn.BatchNorm1d(num_features=self.args.d_model * 2),
            nn.LeakyReLU(0.2),
            # nn.Dropout(p=0.3),
            nn.Linear(
                int(self.args.d_model * 2), int(self.args.d_model * 4), bias=True
            ),
            nn.LeakyReLU(0.2),
            # nn.Dropout(p=0.3),
            nn.Linear(
                int(self.args.d_model * 4), int(self.args.d_model * 4), bias=True
            ),
            nn.LeakyReLU(0.2),
            # nn.Dropout(p=0.3),
            nn.Linear(
                int(self.args.d_model * 4), int(self.args.d_model * 2), bias=True
            ),
            nn.BatchNorm1d(num_features=self.args.d_model * 2),
            nn.LeakyReLU(0.2),
            # nn.Dropout(p=0.3),
            nn.Linear(
                int(self.args.d_model * 2), int(self.args.d_model * 2), bias=True
            ),
            nn.BatchNorm1d(num_features=self.args.d_model * 2),
            nn.LeakyReLU(0.2),
            # nn.Dropout(p=0.3),
            nn.Linear(int(self.args.d_model * 2), int(self.args.d_model), bias=True),
            nn.LeakyReLU(0.2),
            # nn.Dropout(p=0.3),
            nn.Linear(int(self.args.d_model), int(self.args.d_model), bias=True),
            nn.LeakyReLU(0.2),
            # nn.Dropout(p=0.3),
            nn.Linear(int(self.args.d_model), int(self.args.d_model / 2), bias=True),
            nn.LeakyReLU(0.2),
            # nn.Dropout(p=0.3),
            nn.Linear(int(self.args.d_model / 2), self.labels, bias=True),
        )

    @staticmethod
    def init_weights(m):
        """
        initialize nn weights，called in main.py
        :param m: parameter or the nn
        :return:
        """
        if type(m) == torch.nn.Linear:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, mean=0.0, std=0)
        elif type(m) == torch.nn.Embedding:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)

    def forward(self, embeddings, labels):
        output = self.predict(embeddings)["output"]
        if torch.cuda.device_count() > 0:
            labels = labels.cpu().type(torch.LongTensor).cuda()
        else:
            labels = labels.type(torch.LongTensor)
        if output.squeeze().dim() == 0:
            loss = self.criterion(output.unsqueeze(0), labels)
        else:
            loss = self.criterion(output, labels)
        return loss

    def predict(self, embeddings):
        scores = self.network(embeddings)
        output = self.softmax(scores)

        # prediction = output.max(1, keepdim=True)[1]     # get the index of the max
        prediction = torch.argmax(output, dim=1)

        result_dict = {"output": output, "prediction": prediction}
        return result_dict

    """
    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        dir_path = os.path.dirname(model_path)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        torch.save(self.state_dict(), model_path)
        # logging.info('Save ' + self.name + ' discriminator model to ' + model_path)

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        self.load_state_dict(torch.load(model_path))
        self.eval()
        logging.info("Load " + self.name + " discriminator model from " + model_path)
    """
