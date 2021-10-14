import sys
import re
from operator import add
import numpy as np
from numpy.random.mtrand import dirichlet, multinomial
from pyspark import SparkContext

if __name__ == "__main__":
    input_dir = sys.argv[1]  # "20-news-same-line.txt"
    sc = SparkContext(appName="A6")
    lines = sc.textFile(input_dir)
    header_text = lines.map(lambda x: (x[x.index('id=') + 4:x.index('" url=')], x[x.index('">') + 2:][:-8].lower()))
    regex = re.compile('[^a-zA-Z]')


    def get_words(input_val):
        result = []
        words = regex.sub(' ', input_val).split()
        for w in words:
            if len(w) > 2:
                result.append(w)
        return result


    header_words = header_text.map(lambda x: (x[0], get_words(x[1])))
    top_size = 20000
    sorted_words = header_words.flatMap(lambda x: x[1]).map(lambda x: (x, 1)).reduceByKey(add).takeOrdered(top_size,
                                                                                                           key=lambda
                                                                                                               x: -x[1])
    top_words = []
    for each in sorted_words:
        top_words.append(each[0])


    def countWords(input_val):
        header = input_val[0]
        words = input_val[1]
        numwords = np.zeros(top_size)
        count = 0
        for w in words:
            if w in top_words:
                count += 1
                idx = top_words.index(w)
                numwords[idx] += 1
        return (header, numwords, count)


    result = header_words.map(countWords)
    result.cache()

    target = result.filter(lambda x: x[0].split("/")[-1][0:] == "37261").collect()
    sort_index = list(np.argsort(target[0][1]))
    list.reverse(sort_index)

    target_100_words = []
    for i in range(177):
        index = sort_index[i]
        target_100_words.append((top_words[index], target[0][1][index]))
    print("The 100 most common words appear in document 20 newsgroups/comp.graphics/37261:")
    for w in target_100_words:
        print(w)

    alpha = [0.1] * 20
    beta = np.array([0.1] * top_size)

    pi = np.array(dirichlet(alpha).tolist())

    mu = np.array([dirichlet(beta) for j in range(20)])
    log_mu = np.log(mu)


    def getProbs(checkParams, log_allMus, input_val, log_pi):
        topic = input_val[0].split("/")[1]
        x = input_val[1]
        if checkParams == True:
            if x.shape[0] != log_allMus.shape[1]:
                raise Exception('Number of words in doc does not match')
            if log_pi.shape[0] != log_allMus.shape[0]:
                print(log_pi.shape[0])
                raise Exception('Number of document classes does not match')

            if not (0.999 <= np.sum(np.exp(log_pi)) <= 1.001):
                raise Exception('Pi is not a proper probability vector')
            for i in range(log_allMus.shape[0]):
                if not (0.999 <= np.sum(np.exp(log_allMus[i])) <= 1.001):
                    raise Exception('log_allMus[' + str(i) + '] is not a proper probability vector')

        allProbs = np.copy(log_pi)
        for i in range(log_allMus.shape[0]):
            product = np.multiply(x, log_allMus[i])
            allProbs[i] += np.sum(product)
        biggestProb = np.amax(allProbs)
        allProbs -= biggestProb
        allProbs = np.exp(allProbs)
        allProbs = allProbs / np.sum(allProbs)
        return (x, allProbs, topic)


    def get_cat(prob):
        return np.nonzero(multinomial(1, prob))[0][0]


    for num_iter in range(200):
        print("epoch: {}".format(num_iter))
        logPi = np.log(pi)
        x_c_topic = result.map(lambda x_i: getProbs(False, log_mu, x_i, logPi)) \
            .map(lambda x: (x[0], get_cat(x[1]), x[2]))
        x_c_topic.cache()
        c_count = x_c_topic.map(lambda x: (x[1], 1)).reduceByKey(add).sortByKey(ascending=True).collectAsMap()
        # Now, we update the alpha
        new_alpha = [0] * 20
        for i in range(20):
            if i in c_count:
                new_alpha[i] = alpha[i] + c_count[i]
            else:
                new_alpha[i] = alpha[i]

        # use the new alpha to take samples from dirichlet.
        pi = dirichlet(new_alpha)

        empty = sc.parallelize([np.array([0] * top_size)])
        for j in range(20):
            w_count = x_c_topic.filter(lambda term: term[1] == j) \
                .map(lambda term: term[0]) \
                .union(empty) \
                .reduce(np.add)
            log_mu[j] = np.log(dirichlet(np.add(beta, w_count)))

    mu = np.exp(log_mu)
    for cat in range(20):
        mu_i = mu[cat]
        sort_index = list(np.argsort(mu_i))
        list.reverse(sort_index)
        sort_index = sort_index[0:50]
        words = []
        for index in sort_index:
            word = top_words[index]
            words.append(word)
        print("50 important_words for category {} : {}".format(cat, words))

    # Task4
    c_topic = x_c_topic.map(lambda x: (x[1], x[2]))
    c_topic.cache()
    for cat in range(20):
        topic_count = c_topic.filter(lambda x: x[0] == cat).map(lambda x: (x[1], 1)).reduceByKey(add).collect()
        topic_count = [list(x) for x in topic_count]
        topic_count.sort(key=lambda x: -x[1])
        total_count = c_count[cat]
        top_3_pro = topic_count[0:3]
        for each in top_3_pro:
            each[1] = round(float(each[1]) / float(total_count), 4)
        print("Total count of cluster {}: {}".format(cat, total_count))
        print("Top 3 topics in cluster {} : {}".format(cat, top_3_pro))