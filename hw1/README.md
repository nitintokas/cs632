# cs632
Introduction to deep learning

Assignment1:

PART 1:

1. In a Nearest Neighbor classifier, is it important that all features be on the same scale? Think: what would happen if one feature ranges between 0-1, and another ranges between 0-1000? If it is important that they are on the same scale, how could you achieve this?

Answer:
In my view the question about scaling/not scaling the features in machine learning is a statement about the measurement units of your features. And it is related to the prior knowledge you have about the problem. Some of the algorithms, do feature scaling by design and you would have no effect in performing one manually. Others, like knn can be gravely affected by it.

So with knn type of classifier you have to measure the distances between pairs of samples. The distances will of course be influenced by the measurement units one uses. Imagine you are classifying population into males and females and you have a bunch of measurements including height. Now your classification result will be influenced by the measurements the height was reported in. If the height is measured in nanometers then it's likely that any k nearest neighbors will merely have similar measures of height. You have to scale.

However as a contrast example imagine classifying something that has equal units of measurement recorded with noise. Like a photograph or microarray or some spectrum. in this case you already know a-priori that your features have equal units. If you were to scale them all you would amplify the effect of features that are constant across all samples, but were measured with noise. (Like a background of the photo). This again will have an influence on knn and might drastically reduce performance if your data had more noisy constant values compared to the ones that vary. Now any similarity between k nearest neighbors will get influenced by noise.

So this is like with everything else in machine learning - use prior knowledge whenever possible.


2. What is the difference between a numeric and categorical feature? How might you represent a categorical feature so your Nearest Neighbor classifier could work with it?

Answer:
With a numerical feature you can differentiate every single data entry but with categorical feature you can only make a binary difference i.e., either 0 or 1(It exists or it it doesn’t). The use of binary indicator variables solves this problem implicitly. This has the benefit of allowing you to continue your probably matrix based implementation with this kind of data.

There is an infinite number of such combinations. You need to experiment which works best for you. Essentially, you might want to use some classic metric on the numeric values plus a distance on the other attributes, scaled appropriately.


3.	What is the importance of testing data ?

Answer:
A test set is a set of data used to assess the strength and utility of a predictive relationship of the given dataset and make find out that our particular classifier is accurate on the given data or not.


4.	What does “supervised” refer to in “supervised classification”?

Answer:
Supervised refers to having prior knowledge about the problem and then running classification with pre determined training data and features. It is done in a user managed environment with less chances of miscalculation, because what the result should be expected known and hence the classification is made accordingly.

5.	If you were to include additional features for the Iris dataset, what would they be, and why?

Answer:
Iris dataset includes 3 classes: Iris setosa, Iris versicolor and Iris virginica. The further classification is on the basis of sepal-length,sepal-width,petal-length, petal-width. I feel that stigma and color should also be included, i.e., stigma-height as they basically have the same width for most of them but the height of stigma differ completely from one another. Also different types of flowers have different colors(mostly) so it should also be included. Altogether with all these additional features when combined with previous inbuilt features, the accuracy will improve drastically.


...........
PART 2

1. What are the strengths and weaknesses of using a Bag of Words?

Answer: Bag of words ignores the context of words, simplify ignoring th problem at some cost. It can fail badly depending on specific case.

Example:

God Father != Father god

Bag of Words treatment couldn't distinguish these two cases.



2. Which of your features do you think is most predictive, least predictive, and why?
Answer: Features with highest weight are most predictive and common words as least predictive as being part of bag of words.

3. Did your classifier misclassify any examples? Why or why not?
Answer: The presence of tags were a big part of miscalculations, for example, html and xml tags. They confused the parser and hence reduction in accuracy, also data set is very small to train the classifier.
