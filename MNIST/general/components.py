import torch
from torch import nn
import torchsummary
from torch.nn import functional as F
from settings import prototype_shape, n_classes, encoder_out_channels

class EncoderBlock(nn.Module):
    def __init__(self, input_channels=1, latent_channels=prototype_shape[1], ksize=3):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=ksize, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=ksize, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=ksize, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, encoder_out_channels, kernel_size=ksize, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x


class EncoderBottleneck(nn.Module):
    def __init__(self, input_channels=encoder_out_channels, latent_channels=prototype_shape[1], ksize=3):
        super(EncoderBottleneck, self).__init__()
        self.conv_mu = nn.Conv2d(encoder_out_channels, latent_channels, kernel_size=1, stride=1, padding="valid")
        self.conv_logvar = nn.Conv2d(encoder_out_channels, latent_channels, kernel_size=1, stride=1, padding="valid")

    def forward(self, x):
        mu = F.sigmoid(self.conv_mu(x))
        logvar = F.sigmoid(self.conv_logvar(x))
        return mu, logvar


class DecoderBlock(nn.Module):
    def __init__(self, latent_channels=prototype_shape[1], output_channels=1, ksize=3):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(latent_channels, encoder_out_channels, kernel_size=1, stride=1, padding="valid")
        self.deconv1 = nn.ConvTranspose2d(encoder_out_channels, 32, kernel_size=ksize, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size=ksize, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size=ksize, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, output_channels, kernel_size=ksize, stride=2, padding=1,  output_padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))  # Using sigmoid for image reconstruction
        return x


class ClassifierBlock(nn.Module):
    def __init__(self, prototype_activation_function='log'):
        super(ClassifierBlock, self).__init__()
        self.prototype_activation_function = prototype_activation_function
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.n_classes = n_classes
        self.epsilon = 1e-4

        '''
        Initialize prototype vectors. This was obtained from ProtoPNet
        '''
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                              requires_grad=True)

        self.ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)

        self.last_layer = nn.Linear(self.num_prototypes, self.n_classes,
                                    bias=False)  # do not use bias

        # Initialize weights for the last layer
        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)

        def _l2_convolution(self, x):

            # apply self.prototype_vectors as l2-convolution filters on input x
            x2 = x ** 2
            x2_patch_sum = F.conv2d(input=x2, weight=self.ones)

            p2 = self.prototype_vectors ** 2
            p2 = torch.sum(p2, dim=(1, 2, 3))
            # p2 is a vector of shape (num_prototypes,)
            # then we reshape it to (num_prototypes, 1, 1)
            p2_reshape = p2.view(-1, 1, 1)

            xp = F.conv2d(input=x, weight=self.prototype_vectors)
            intermediate_result = - 2 * xp + p2_reshape  # use broadcast
            # x2_patch_sum and intermediate_result are of the same shape
            distances = F.relu(x2_patch_sum + intermediate_result)

            return distances

        def prototype_distances(self, x):

            # x is the sample from bottleneck
            distances = self._l2_convolution(x)
            return distances

        def distance_2_similarity(self, distances):
            if self.prototype_activation_function == 'log':
                return torch.log((distances + 1) / (distances + self.epsilon))
            elif self.prototype_activation_function == 'linear':
                return -distances
            else:
                return self.prototype_activation_function(distances)


        def set_last_layer_incorrect_connection(self, incorrect_strength):
            '''
            the incorrect strength will be actual strength if -0.5 then input -0.5
            '''
            positive_one_weights_locations = torch.t(self.prototype_class_identity)
            negative_one_weights_locations = 1 - positive_one_weights_locations

            correct_class_connection = 1
            incorrect_class_connection = incorrect_strength
            self.last_layer.weight.data.copy_(
                correct_class_connection * positive_one_weights_locations
                + incorrect_class_connection * negative_one_weights_locations)

        def forward(self, x):
            distances = self.prototype_distances(x)
            '''
            we cannot refactor the lines below for similarity scores
            because we need to return min_distances
            '''
            # global min pooling because min distance corresponds to max similarity
            min_distances = -F.max_pool2d(-distances, kernel_size=(distances.size()[2],distances.size()[3]))
            min_distances = min_distances.view(-1, self.num_prototypes)
            prototype_activations = self.distance_2_similarity(min_distances)
            logits = self.last_layer(prototype_activations)
            return logits, min_distances


if __name__ == '__main__':
    encoder = EncoderBlock()
    print(encoder)
    print(torchsummary.summary(encoder, (1, 28, 28)))

    decoder = DecoderBlock()
    print(decoder)
    print(torchsummary.summary(decoder, (10, 2, 2)))

    classifier = ClassifierBlock()
    print(classifier)
    print(torchsummary.summary(classifier, (10, 2, 2)))

