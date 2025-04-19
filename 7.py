from fastai.collab import *
from fastai.tabular.all import *
import os

# if __name__ == '__main__':
set_seed(42)
path = untar_data(URLs.ML_100k)

# prepare data
ratings = pd.read_csv(path/'u.data', delimiter='\t', header=None,
                      names=['user', 'movie', 'rating', 'timestamp'])
movies = pd.read_csv(path/'u.item', delimiter='|', encoding='latin-1', header=None,
                     names=('movie', 'title'), usecols=(0, 1))
ratings = ratings.merge(movies)

# construct model
dls = CollabDataLoaders.from_df(ratings, item_name='title', bs=64)

n_users = len(dls.classes['user'])
n_movies = len(dls.classes['title'])
n_factors = 50

user_factors = torch.randn(n_users, n_factors)
movie_factors = torch.randn(n_movies, n_factors)


y_range_which_make_sense = (1, 6)
y_range_from_book = (0, 5.5)


class DotProductBias_(Module):
    # built-in Embedding
    def __init__(self,
                 n_users, n_movies, n_factors,
                 y_range=y_range_which_make_sense):
        #  y_range=y_range_from_book):

        self.user_factors = Embedding(n_users, n_factors)
        self.movie_factors = Embedding(n_movies, n_factors)

        self.user_bias = Embedding(n_users, 1)
        self.movie_bias = Embedding(n_movies, 1)

        self.y_range = y_range

    def forward(self, x):
        users = self.user_factors(x[:, 0])
        movies = self.movie_factors(x[:, 1])
        res = (users * movies).sum(dim=1, keepdim=True)
        res += self.user_bias(x[:, 0]) + self.movie_bias(x[:, 1])
        return sigmoid_range(res, *self.y_range)


def create_params(size):
    return nn.Parameter(torch.zeros(*size).normal_(0, 0.01))


class DotProductBias(Module):
    # Our own embedding
    def __init__(self,
                 n_users, n_movies, n_factors,
                 y_range=y_range_which_make_sense):
        #  y_range=y_range_from_book):

        self.user_factors = create_params([n_users, n_factors])
        self.movie_factors = create_params([n_movies, n_factors])

        self.user_bias = create_params([n_users])
        self.movie_bias = create_params([n_movies])

        self.y_range = y_range

    def forward(self, x):
        users = self.user_factors[x[:, 0]]
        movies = self.movie_factors[x[:, 1]]
        res = (users * movies).sum(dim=1)
        res += self.user_bias[x[:, 0]] + self.movie_bias[x[:, 1]]
        return sigmoid_range(res, *self.y_range)


model = DotProductBias(n_users, n_movies, n_factors)
learn = Learner(dls, model, loss_func=MSELossFlat())
retrain = False

MODEL_PATH = path/'models/dot_model.pth'
os.makedirs(path/'models', exist_ok=True)
if MODEL_PATH.exists() and not retrain:
    state = torch.load(MODEL_PATH)
    learn.model.load_state_dict(state)
    print("Loaded saved model—skipping training.")
else:
    learn.fit_one_cycle(5, 5e-3, wd=0.1)
    torch.save(learn.model.state_dict(), MODEL_PATH)
    print("Trained and saved model.")

movie_bias = learn.model.movie_bias.squeeze()
idxs = movie_bias.argsort(descending=True)[:15]
classics = [dls.classes['title'][i] for i in idxs]

print(classics)
