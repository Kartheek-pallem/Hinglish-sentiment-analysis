              model features            comparison    change    low_95   high_95        real_change
      Decision tree     char        raw -> cleaned  0.012296 -0.008145  0.032319 no (could be luck)
      Decision tree     char cleaned -> normalized -0.027532 -0.045330 -0.008596                yes
      Decision tree     char     raw -> normalized -0.015236 -0.036127  0.005320 no (could be luck)
      Decision tree     word        raw -> cleaned  0.005262 -0.015063  0.024056 no (could be luck)
      Decision tree     word cleaned -> normalized  0.019824 -0.001591  0.040086 no (could be luck)
      Decision tree     word     raw -> normalized  0.025086  0.004941  0.045943                yes
         Linear SVM     char        raw -> cleaned  0.008328 -0.003005  0.019522 no (could be luck)
         Linear SVM     char cleaned -> normalized -0.000433 -0.007051  0.006963 no (could be luck)
         Linear SVM     char     raw -> normalized  0.007895 -0.003640  0.020752 no (could be luck)
         Linear SVM     word        raw -> cleaned -0.006425 -0.015478  0.002306 no (could be luck)
         Linear SVM     word cleaned -> normalized  0.007967  0.000528  0.015532                yes
         Linear SVM     word     raw -> normalized  0.001541 -0.008497  0.011974 no (could be luck)
Logistic regression     char        raw -> cleaned  0.012256 -0.000716  0.025868 no (could be luck)
Logistic regression     char cleaned -> normalized  0.006588 -0.003154  0.016310 no (could be luck)
Logistic regression     char     raw -> normalized  0.018845  0.007065  0.029771                yes
Logistic regression     word        raw -> cleaned -0.009435 -0.019784  0.001477 no (could be luck)
Logistic regression     word cleaned -> normalized  0.001821 -0.005161  0.009257 no (could be luck)
Logistic regression     word     raw -> normalized -0.007614 -0.018192  0.002978 no (could be luck)
        Naive Bayes     char        raw -> cleaned  0.016864  0.003356  0.029829                yes
        Naive Bayes     char cleaned -> normalized  0.002362 -0.002367  0.007361 no (could be luck)
        Naive Bayes     char     raw -> normalized  0.019226  0.005797  0.032199                yes
        Naive Bayes     word        raw -> cleaned  0.003763 -0.005031  0.012557 no (could be luck)
        Naive Bayes     word cleaned -> normalized  0.003740 -0.003612  0.010583 no (could be luck)
        Naive Bayes     word     raw -> normalized  0.007504 -0.002754  0.017739 no (could be luck)
