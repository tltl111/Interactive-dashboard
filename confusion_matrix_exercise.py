import numpy

actual = numpy.random.binomial(1, 0.99, size=1000)
predicted = numpy.random.binomial(1, 0.5, size=1000)

from sklearn import metrics

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                            display_labels=[False, True])
import matplotlib.pyplot as plt

cm_display.plot()
plt.show()