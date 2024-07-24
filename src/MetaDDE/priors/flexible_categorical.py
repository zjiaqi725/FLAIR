import time
import random

import torch
from torch import nn

from .utils import get_batch_to_dataloader
from tabpfn.utils import normalize_data, nan_handling_missing_for_unknown_reason_value, nan_handling_missing_for_no_reason_value, nan_handling_missing_for_a_reason_value, to_ranking_low_mem, remove_outliers, normalize_by_used_features_f
from .utils import randomize_classes, CategoricalActivation
from .utils import uniform_int_sampler_f
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore", category=Warning)
time_it = False

class BalancedBinarize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (x > torch.median(x)).float()

def class_sampler_f(min_, max_):
    def s():
        if random.random() > 0.5:
            return uniform_int_sampler_f(min_, max_)()
        return 2
    return s

class RegressionNormalized(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x has shape (T,B)
        
        # TODO: Normalize to -1, 1 or gaussian normal
        maxima = torch.max(x, 0)[0]
        minima = torch.min(x, 0)[0]
        minima = torch.where(maxima == minima, maxima - 1, minima)
        # Normalize to [0, 1] range
        norm = (x - minima) / (maxima-minima)

        return norm

class MulticlassRank(nn.Module):
    def __init__(self, num_classes, ordered_p=0.5):
        super().__init__()
        self.num_classes = class_sampler_f(2, num_classes)()
        self.ordered_p = ordered_p

    def forward(self, x):
        # x has shape (T,B,H)

        # CAUTION: This samples the same idx in sequence for each class boundary in a batch
        class_boundaries = torch.randint(0, x.shape[0], (self.num_classes - 1,))
        class_boundaries = x[class_boundaries].unsqueeze(1)

        d = (x > class_boundaries).sum(axis=0)

        randomized_classes = torch.rand((d.shape[1], )) > self.ordered_p
        d[:, randomized_classes] = randomize_classes(d[:, randomized_classes], self.num_classes)
        reverse_classes = torch.rand((d.shape[1],)) > 0.5
        d[:, reverse_classes] = self.num_classes - 1 - d[:, reverse_classes]
        return d

class MulticlassValue(nn.Module):
    def __init__(self, num_classes, ordered_p=0.5):
        super().__init__()
        self.num_classes = class_sampler_f(2, num_classes)()
        self.classes = nn.Parameter(torch.randn(self.num_classes-1), requires_grad=False)
        self.ordered_p = ordered_p

    def forward(self, x):
        # x has shape (T,B,H)
        d = (x > (self.classes.unsqueeze(-1).unsqueeze(-1))).sum(axis=0)

        randomized_classes = torch.rand((d.shape[1],)) > self.ordered_p
        d[:, randomized_classes] = randomize_classes(d[:, randomized_classes], self.num_classes)
        reverse_classes = torch.rand((d.shape[1],)) > 0.5
        d[:, reverse_classes] = self.num_classes - 1 - d[:, reverse_classes]
        return d

class MulticlassMultiNode(nn.Module):
    def __init__(self, num_classes, ordered_p=0.5):
        super().__init__()
        self.num_classes = class_sampler_f(2, num_classes)()
        self.classes = nn.Parameter(torch.randn(num_classes-1), requires_grad=False)
        self.alt_multi_class = MulticlassValue(num_classes, ordered_p)

    def forward(self, x):
        # x has shape T, B, H
        if len(x.shape) == 2:
            return self.alt_multi_class(x)
        T = 3
        x[torch.isnan(x)] = 0.00001
        d = torch.multinomial(torch.pow(0.00001+torch.sigmoid(x[:, :, 0:self.num_classes]).reshape(-1, self.num_classes), T), 1, replacement=True).reshape(x.shape[0], x.shape[1])#.float()
        return d


class FlexibleCategorical(torch.nn.Module):
    def __init__(self, get_batch, hyperparameters, args):
        super(FlexibleCategorical, self).__init__()

        self.h = {k: hyperparameters[k]() if callable(hyperparameters[k]) else hyperparameters[k] for k in
                                hyperparameters.keys()}
        self.args = args
        self.args_passed = {**self.args}
        self.args_passed.update({'num_features': self.h['num_features_used']})
        self.get_batch = get_batch

        warnings.filterwarnings("ignore", category=Warning)
        if self.h['num_classes'] == 0:
            self.class_assigner = RegressionNormalized()
        else:
            if self.h['num_classes'] > 1 and not self.h['balanced']:
                if self.h['multiclass_type'] == 'rank':
                    self.class_assigner = MulticlassRank(self.h['num_classes']
                                                 , ordered_p=self.h['output_multiclass_ordered_p']
                                                 )
                elif self.h['multiclass_type'] == 'value':
                    self.class_assigner = MulticlassValue(self.h['num_classes']
                                                         , ordered_p=self.h['output_multiclass_ordered_p']
                                                         )
                elif self.h['multiclass_type'] == 'multi_node':
                    self.class_assigner = MulticlassMultiNode(self.h['num_classes'])
                else:
                    raise ValueError("Unknow Multiclass type")
            elif self.h['num_classes'] == 2 and self.h['balanced']:
                self.class_assigner = BalancedBinarize()
            elif self.h['num_classes'] > 2 and self.h['balanced']:
                raise NotImplementedError("Balanced multiclass training is not possible")

    def drop_for_reason(self, x, v):
        nan_prob_sampler = CategoricalActivation(ordered_p=0.0
                                                 , categorical_p=1.0
                                                 , keep_activation_size=False,
                                                 num_classes_sampler=lambda: 20)
        d = nan_prob_sampler(x)
        # TODO: Make a different ordering for each activation
        x[d < torch.rand((1,), device=x.device) * 20 * self.h['nan_prob_no_reason'] * random.random()] = v
        return x

    def drop_for_no_reason(self, x, v):
        x[torch.rand(x.shape, device=self.args['device']) < random.random() * self.h['nan_prob_no_reason']] = v
        return x

    def forward(self, batch_size):
        start = time.time()
        x, y, y_ = self.get_batch(hyperparameters=self.h, **self.args_passed)
        # print('self.get_batch', self.get_batch)
        # print('flexible_categorical_x' , x)
        # print('x.shape', x.shape)  #(seq_len, batch_size, num_features)
        # print('flexible_categorical_y' , y)
        # print('y.shape', y.shape)
        
        if torch.isnan(y).sum() > 0:
            print('initially Nans in target!')
        if time_it:
            print('Flex Forward Block 1', round(time.time() - start, 3))

        start = time.time()

        if self.h['nan_prob_no_reason']+self.h['nan_prob_a_reason']+self.h['nan_prob_unknown_reason'] > 0 and random.random() > 0.5: # Only one out of two datasets should have nans
            # print('!!nan_prob!!')
            if random.random() < self.h['nan_prob_no_reason']: # Missing for no reason
                x = self.drop_for_no_reason(x, nan_handling_missing_for_no_reason_value(self.h['set_value_to_nan']))

            if self.h['nan_prob_a_reason'] > 0 and random.random() > 0.5: # Missing for a reason
                x = self.drop_for_reason(x, nan_handling_missing_for_a_reason_value(self.h['set_value_to_nan']))

            if self.h['nan_prob_unknown_reason'] > 0: # Missing for unknown reason  and random.random() > 0.5
                if random.random() < self.h['nan_prob_unknown_reason_reason_prior']:
                    x = self.drop_for_no_reason(x, nan_handling_missing_for_unknown_reason_value(self.h['set_value_to_nan']))
                else:
                    x = self.drop_for_reason(x, nan_handling_missing_for_unknown_reason_value(self.h['set_value_to_nan']))
        
    
        # Categorical features
        if 'categorical_feature_p' in self.h and random.random() < self.h['categorical_feature_p']:
            # print('!!categorical_feature!!', 'x_col_total', x.shape[2])  #self.h['categorical_feature_p']=0.2
            cat_feature_list = []
            p = random.random()
            for col in range(x.shape[2]):
                num_unique_features = max(round(random.gammavariate(1,10)),2)
                m = MulticlassRank(num_unique_features, ordered_p=0.3)
                if random.random() < p:
                    cat_feature_list.append(col)
                    x[:, :, col] = m(x[:, :, col])
            # print('cat_feature_list', cat_feature_list)
            # print('before_x.shape', x.shape)
            # One-hot encode categorical features
            if len(cat_feature_list) != 0 and self.h['onehot']:
                # Separate numerical and categorical features
                numer_features = torch.tensor(x[:,:, [i for i in range(x.shape[2]) if i not in cat_feature_list]], dtype=torch.float32)
                cat_features = torch.tensor(x[:,:, cat_feature_list].cpu().detach(), dtype=torch.int64) if cat_feature_list else None
                feature_encoder = OneHotEncoder(sparse=False)
                assert len(cat_features.shape)==3
                cat_features = cat_features.reshape(-1,cat_features.shape[-1])
                # print('cat_features', cat_features.shape)
                cat_features_encoded = feature_encoder.fit_transform(cat_features)
                if cat_features_encoded.shape[-1] > (self.args['num_features']-numer_features.shape[-1]):
                    cat_num_features = self.args['num_features']-numer_features.shape[-1]
                    pca = PCA(n_components=cat_num_features)
                    cat_features_encoded = pca.fit_transform(cat_features_encoded)
                # print('cat_features_encoded1', cat_features_encoded.shape)
                cat_features_encoded = cat_features_encoded.reshape(numer_features.shape[0],numer_features.shape[1] ,-1)
                
                # cat_features_encoded = []
                # for a in range(x.shape[1]):
                #     cat_features_a = cat_features[:,a,:]
                #     cat_features_encoded_a = feature_encoder.fit_transform(cat_features_a)
                #     print('cat_features_encoded_a1', cat_features_encoded_a.shape, numer_features.shape)
                #     if cat_features_encoded_a.shape[-1] > (self.args['num_features']-numer_features.shape[-1]):
                #         cat_num_features = self.args['num_features']-numer_features.shape[-1]
                #         # print('cat_num_features', cat_num_features)
                #         pca = PCA(n_components=cat_num_features)
                #         cat_features_encoded_a = pca.fit_transform(cat_features_encoded_a)
                #     print('cat_features_encoded_a2', cat_features_encoded_a.shape)
                #     cat_features_encoded.append(torch.tensor(cat_features_encoded_a, device=self.args['device']).unsqueeze(1))
                # cat_features_encoded = torch.cat(cat_features_encoded,dim=1)
                # print('cat_features_encoded2', cat_features_encoded.shape)
                
                # print('cat_features_encoded', cat_features_encoded.shape, numer_features.shape)    
                # Combine numerical and encoded categorical features
                x = torch.tensor(
                    torch.cat((numer_features, torch.tensor(cat_features_encoded, device=self.args['device'])), dim=-1),
                    dtype=torch.float32
                )
                # print('OneHot_cat_features_encoded' , cat_features_encoded.shape, cat_features.shape)
                # print('OneHot_x' , x, x.shape)                    
                self.h['num_features_used'] = x.shape[2]
                
        if time_it:
            print('Flex Forward Block 2', round(time.time() - start, 3))
            start = time.time()

        if self.h['normalize_to_ranking']:
            x = to_ranking_low_mem(x)
        else:
            x = remove_outliers(x)
        x, y = normalize_data(x), normalize_data(y)
        # print('flexible_categorical_x' , x)
        # print('normalize_x.shape', x.shape)
        # print('flexible_categorical_y' , y)
        # print('y.shape', y.shape)
        
        if time_it:
            print('Flex Forward Block 3', round(time.time() - start, 3))
            start = time.time()

        # Cast to classification if enabled
        # y = self.class_assigner(y).float()
        # print('normal', y)

        if time_it:
            print('Flex Forward Block 4', round(time.time() - start, 3))
            start = time.time()
        if self.h['normalize_by_used_features']:
            x = normalize_by_used_features_f(x, self.h['num_features_used'], 
                                             self.args['num_features'], 
                                             normalize_with_sqrt=self.h.get('normalize_with_sqrt',False))
        if time_it:
            print('Flex Forward Block 5', round(time.time() - start, 3))

        start = time.time()
        # Append empty features if enabled
        if x.shape[2] < self.args['num_features']:
            x = torch.cat(
                [x, torch.zeros((x.shape[0], x.shape[1], self.args['num_features'] - self.h['num_features_used']),
                            device=self.args['device'])], -1)
        # print('cat_features', x.shape)
        if time_it:
            print('Flex Forward Block 6', round(time.time() - start, 3))

        if torch.isnan(y).sum() > 0:
            print('Nans in target!')

        if self.h['check_is_compatible'] and (self.h['num_classes'] != 0 ):
            for b in range(y.shape[1]):
                is_compatible, N = False, 0
                while not is_compatible and N < 10:
                    targets_in_train = torch.unique(y[:self.args['single_eval_pos'], b], sorted=True)
                    targets_in_eval = torch.unique(y[self.args['single_eval_pos']:, b], sorted=True)

                    is_compatible = len(targets_in_train) == len(targets_in_eval) and (
                                targets_in_train == targets_in_eval).all() and len(targets_in_train) > 1

                    if not is_compatible:
                        randperm = torch.randperm(x.shape[0])
                        x[:, b], y[:, b] = x[randperm, b], y[randperm, b]
                    N = N + 1
                if not is_compatible:
                    if not is_compatible:
                        # todo check that it really does this and how many together
                        y[:, b] = -100 # Relies on CE having `ignore_index` set to -100 (default)

        if self.h['normalize_labels']and (self.h['num_classes'] != 0 ):
            #assert self.h['output_multiclass_ordered_p'] == 0., "normalize_labels destroys ordering of labels anyways."
            for b in range(y.shape[1]):
                valid_labels = y[:,b] != -100
                if self.h.get('normalize_ignore_label_too', False):
                    valid_labels[:] = True
                y[valid_labels, b] = (y[valid_labels, b] > y[valid_labels, b].unique().unsqueeze(1)).sum(axis=0).unsqueeze(0).float()

                if y[valid_labels, b].numel() != 0 and self.h.get('rotate_normalized_labels', True):
                    num_classes_float = (y[valid_labels, b].max() + 1).cpu()
                    num_classes = num_classes_float.int().item()
                    assert num_classes == num_classes_float.item()
                    random_shift = torch.randint(0, num_classes, (1,), device=self.args['device'])
                    y[valid_labels, b] = (y[valid_labels, b] + random_shift) % num_classes
        # print('flexible_categorical_y222' , y)
        return x, y, y  # x.shape = (T,B,H)

import torch.cuda as cutorch

@torch.no_grad()
def get_batch(batch_size, seq_len, num_features, get_batch, device, hyperparameters=None, batch_size_per_gp_sample=None, **kwargs):
    batch_size_per_gp_sample = batch_size_per_gp_sample or (min(32, batch_size))
    num_models = batch_size // batch_size_per_gp_sample
    assert num_models > 0, f'Batch size ({batch_size}) is too small for batch_size_per_gp_sample ({batch_size_per_gp_sample})'
    assert num_models * batch_size_per_gp_sample == batch_size, f'Batch size ({batch_size}) not divisible by batch_size_per_gp_sample ({batch_size_per_gp_sample})'

    # Sample one seq_len for entire batch
    seq_len = hyperparameters['seq_len_used']() if callable(hyperparameters['seq_len_used']) else seq_len

    args = {'device': device, 'seq_len': seq_len, 'num_features': num_features, 'batch_size': batch_size_per_gp_sample, **kwargs}

    models = [FlexibleCategorical(get_batch, hyperparameters, args).to(device) for _ in range(num_models)]

    sample = [model(batch_size=batch_size_per_gp_sample) for model in models]

    x, y, y_ = zip(*sample)
    x, y, y_ = torch.cat(x, 1).detach(), torch.cat(y, 1).detach(), torch.cat(y_, 1).detach()
    # print('get_batch', y, y_)
    return x, y, y_

# num_features_used = num_features_used_sampler()
# prior_outputscale = prior_outputscale_sampler()
# prior_lengthscale = prior_lengthscale_sampler()
#
# x, sample = normalize_data(x), normalize_data(sample)
#
# if is_binary_classification:
#     sample = (sample > torch.median(sample, dim=0)[0]).float()
#
# if normalize_by_used_features:
#     x = normalize_by_used_features_f(x, num_features_used, num_features)
#
# # # if is_binary_classification and order_y:
# # #     x, sample = order_by_y(x, sample)
# #
# # Append empty features if enabled
# x = torch.cat([x, torch.zeros((x.shape[0], x.shape[1], num_features - num_features_used), device=device)], -1)

DataLoader = get_batch_to_dataloader(get_batch)