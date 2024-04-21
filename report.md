## COMP3314 Asm 3 

### Background and brief analysis of dataset
In this challenge, we need to create a model that classify images into multiple classes. This dataset contains $50000$ labelled images with a dimension of $32 \times 32$. Each image is classified into one of the ten classes. While the smaller image size means that there are less dimensions to worry about, it also means that some important features might have been lost when the image size is shrunk. Therefore, we need to extract useful features in pre-processing to achieve the best result.

In this challenge, we need to create a model that classify images into multiple classes. This dataset contains 50k labelled images with a dimension of $32 \times 32$. Each image is classified into one of the ten classes. While the smaller image size means that there are less dimensions to worry about, it also means that some important features might have been lost when the image size is shrunk. Therefore, we need to extract useful features in pre-processing to achieve the best result.

### Models

#### Model 1 (Based solely on HOG features)

When doing research on image classification, we see HOG (Histogram of Gradients) is widely used and is capable of providing promising results. We used both `sklearn-image` and `cv2` to generate the HOG features, and we see `cv2` generate the features significantly faster. 

After generating the HOG features, instead of directly feeding into PCA and the SVM like many do, we decided to take a different approach. We notice that we can use LDA (Linear Discriminant Analysis), a supervised learning model, to encode the features into a fewer dimensions and pass onto the SVM to perform classification. As LDA trains to cluster the same labeled data and isolate those with different labels, we can get a much better clustering on the data which makes it easier for the classifier to classify the data. However a limitation of LDA is that it cannot transform the featues into higher dimensions than `n_classes - 1`, in this case, the maximum dimensions that LDA can tranform to is 9. Hence, instead of passing the HOG features to LDA then to SVM, we reduce the dimensions using PCA, and we use LDA to provide "extra" information on the data, which when tested, we see significant boost in the accuracy of the SVM. 

Moreover, instead of simply using `lda.transform` to encode the data, we first computed the centroids of each class, then computed the inverse distance to each of the centroids, hence for those close to a center of a class, they are projected far away from the origin, making classification easier. We also tested this method, and also saw some increase in accuracy.

Lastly, we decided to blur the original image to 4 by 4 images, then concatenated alongisde with the HOG features, befor feeding into PCA and LDA. The idea behind is different classes have different background colors, like planes and boats have bluer background, while deers and birds have generally greener background, hence aiding the model to classify. However we only see some minor increase in accuracy.

In general, the following is our model utilizing HOG Features: 

\<inserts diagram here>

For the fine tuning of parameters, we arbitarily set PCA to reduce to 160 dimensions, while other parmaters are fine tuned with methods explained at the end.

#### Model 2 (Based on K-Means feature learning)

We employ a technique known as patch-based feature extraction to analyze images (\<paper-link>). This method involves dividing each image into smaller, overlapping sections, called "patches". These 6x6 patches are then used as input for our machine learning models, allowing us to capture local information about each image, such as textures or shapes, that might be lost if we were to analyze the image as a whole. We identify this method as we conduct our research on image pre-processing by reading various academic paper.

We whiten and standardize each patches, and then use a KMeans clustering algorithm to further process these patches. The KMeans algorithm groups similar patches together, forming clusters. Each cluster represents a common feature found in our images. The algorithm assigns each patch to the cluster that it most closely resembles, effectively transforming our original image data into a new set of features based on the presence of these common elements, similar to dictionary learning.

Once we have our new feature set, we can use it to train a Support Vector Machine (SVM) classifier. The SVM model is a powerful machine learning algorithm that can classify data into different categories. In our case, it will classify images based on the features extracted from the patches. 

The Support Vector Machine we have chosen is the LinearSVC. It allows faster calculation on high-dimensional data, particularly useful for our data input. Our data input is susceptible to dimensionality reduction, in which a slightly decrease in dimension leads to a sharp drop in prediction accuracy. We also find that LinearSVC performs better than kernel SVC using the input from KMeans algorithm.

However, as the data dimension is very large of size at least 50000 x 3200, this method costs hours of training to secure a relatively satisfactory result. 

#### Model 3 (Based mostly on HOG, Daisy, EOH)

After testing out the previous 2 models, we decided to improve on the HOG model, as training it takes less time and thus allowed for more experiments and improvements. Hence, spending several days testing different features, we agreed on the following features:
- HOG, we found this feature to be extremely useful everytime
- Daisy, throughout testing, we found that this provided significant increase in accuracies on top of HOG
- EOH, a suggestion from a friend, we tested and saw some increase in testing accuracy, though not as significant as Daisy, but since it only adds a handful of features
- Blurring, as from our original model, we see some significant increase in testing accuracy, thus we have kept it

Then, we applied our LDA Encoder on the features which are flattened and concatenated, appending the 10 distance encodings to the features. With some fine tuning, the feature vector is 5131 dimensions, and we used SVM with an rbf kernel to do the classification. With this and some more additional methods, which are discussed at the end in Methodology, we are able to achieve $80\%$ testing accuracy, which is very impressive for a medium-sized model using manually defined features.

#### Model 4 (Similar to Model 3)

While testing different features, we have achieved (where `train_test_split=0.2` and without augmentation) $75\%$ testing accuracy with a very small model, notably having only 250 features, consisting of HOG, Daisy, the blurred out image, with dimensions reduced to 240 using PCA, and 10 extra features being the LDA encodings, which is standardized and passed through a rbf svm. This model is significant in its size, while still maintaining very high accuracy, higher than a lot of those ones found online, like this model that uses HOG, achieving $70\%$ with 3000 features or this model using the $k$-means encoding explained in Model 2 with $k=800$, corresponding to a feature dimension of 3200, which achieved a similar accuracy of $75\%$. 

Mainly, this model is trained fully on a cpu, with some acceleration provided by `scikit-learn-intelex`, it is trained in 44 minutes, which preprocessing took around 15 minutes. While the larger model, Model 3, is trained with HKU Innowing's RTX 4090s, which took 3 minutes.

#### Model 5 (LDA)

This model, though not as good as the others that are showcased, ended up being very useful, especially to the Fine-tuning of the feature extractors. The basic idea behind is due to the supervised learning nature of LDA and the goal of clustering the same class labels and distancing the classes of different labels, it has proven to be very good for extra features, or "encodings", which despite being a few more, is very helpful to the classification model. However, we also extended the functionality of the LDA encoder to allow predictions, where when fitting the model computes the centroids of each clusters by taking the mean encodings of each datapoint of that class, then predicts new points solely on which centroid it is closest to. With the HOG features, this is able to achieve an impressive $63\%$ accuracy, and with the encodings from Model 3, with HOG, Daisy, EOH, and Blurring, we are able to achieve an impressive $69\%$ accuracy. 

We are clear with its drawbacks, namely it does not take the clustering standard deviation into account and assumes each cluster has the same standard deviation. Nonetheless, it has shown its remarkable potential, especially without accelerated learning, where training a new svm model based on some parameters for the feature extractors takes hours. As the accuracy of this model's performance is based on how clustered each class cluster is, which the more clustered the better the final model might have, when fine tuning, we used this model, which takes less than 10 minutes to train, to determine the optimal or close-to-optimal parameters. This allows extremely fast fine-tuning of the features without loosing much accuracy. 

#### Feature extraction based on LBP

Further research on image classification suggests that LBP (Local Binary Pattern) is another potential candidate for object recognition. We used sklearn-image to generate the LBP features. Since LBP works on greyscale images, we first convert the images to greyscale before generating the LBP features.

The rest of the model is similar to our HOG-based model. We use LDA (Linear Discriminant Analysis) and PCA to encode the features into fewer dimensions. After that, we pass it to SVM for classification.

To fine-tune the parameters of LBP, we ran a brute-force search and used the LDA classifier to get an estimation on its final performance. Here is a diagram demonstrating the performances of the LDA classifier under different parameters.

<inserts image here>

We may notice that the accuracy of our LBP model is at around ($30\%~33%$), which is much lower than the HOG-based model. Furthermore,even after concatinating the LBP features to our training data and running SVM, we have only obtained an accuracy of $35.8%$.

If we visualize the image generated by LBP, we may notice that for some images, the outline is barely recognizable. This makes it very hard to classify the image accurately. 

<inserts image here>

This can be due to two reasons: Firstly, LBP only works on images converted to greyscale. As a result, some colour-related features have been lost when the image is converted to greyscale; Secondly, the dimensions of the images are only $32 \times 32$. This might be too low to extract useful features as the images are very blurry and the borders of some objects are not well-defined at all.

It was suggested that combining HOG and LBP will increase the accuracy of the model. Thus, we have also tested a model that uses both HOG and LBP features. However, the accuracy of the model remains about the same ($66\%$). So, based on our findings, LBP might not be a suitable descriptor in this case.

### Methodology 

#### Fine-tuning of Regularization Parameter

Instead of using Grid Search, we propose to use a faster algorithm adapted from Numerical Optimization, the Golden Section Method. As we notice that using SVM, we only have one parameter to tune, which is $C$, and we hypothesis that the function $f(C)$, which gives the mean accuracy for some $C$, is concave. Hence, adapting from the Golden Section Method, we propose the following algorithm to find tune: 

Given a target precision of $\epsilon$, and an intial interval $[a_1, b_1]$, where we know the optimal hyperparameter exists in, we let $\phi = \frac{\sqrt{5}-1}{2}$, then let $$\begin{aligned} \lambda_1 &= a_1 + (1-\phi)(b-a) \\ \mu_1 &= a_1 + \phi(b-a)  \end{aligned}$$ Then, we execute the following until the target precision is reached, i.e. $b_k - a_k < \epsilon$. 
- If $f(\lambda_k) > f(\mu_k)$, then we set $a_{k+1} = a_k$, $b_{k+1} = \mu_k$, $\lambda_{k+1} = a_{k+1} + (1-\phi)(b_{k+1}-a_{k+1})$, and $\mu_{k+1} = \lambda_k$.
- Otherwise, we set $a_{k+1} = \lambda_k$, $b_{k+1} = b_k$, $\lambda_{k+1} = \mu_k$, and $\mu_{k+1} = a_{k+1} + \phi(b_{k+1}-a_{k+1})$. 

After iterating, we let $C^* = \frac{a_n+b_n}{2}$ to be the optimal parameter. As by induction, it is quite easy to show that $\frac{b_k - a_k}{b_{k+1} - a_{k+1}} = \phi$ for any $1 \leq k \leq n-1$, hence to achieve a certain accuracy $\epsilon$, $1 + \log _{\phi} \frac{\epsilon}{b_1 - a_1}$ iterations is sufficient. 

However, we acknowledge that the parameter-to-accuracy curve is not concave, especially around the maximum, hence we use this algorithm to shrink our search to a small enough interval, then perform Grid Search on that interval.

In fact, when fine tuning the regularization parameter $C$ of the HOG model (first model), we are able to increase our accuracy by $2\%$, from $70.24\%$ to over $72\%$, this is done by fine tuning $C$ to 2 decimal places, which took 30 minutes. 

#### Fine-tuning of Feature Extractor Parameters

As in Model 5, we have described an extremely lightweight model for this task, using it, we are able to look through a lot of parameter combinations, and brute-force our way. However, unlike the regularization parameter, we did not make use of that algorithm, as we have noticed that the parameter-to-accuracy curve is not concave, but has quite some noise in it, and that pretty much all parameters are discrete and only span a small range, hence brute-force is highly feasible. 

### Final Model

From all the models, we have selected the best performing one, which is Model 3, and applied augmentation, which in this case would be flipping sideways, to increase our dataset from 50k to 100k. Augmentation has shown to have very significant impact on the final accuracy, where when performing augmentation on Model 3, we say a whopping $4.3\%$ increase in testing accuracy ($75.73\% \to 80.04\%$).