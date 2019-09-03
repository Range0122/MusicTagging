import numpy as np

labels = ['hiphop', 'disco', 'country', 'classical', 'blues', 'reggae', 'rock', 'jazz', 'metal', 'pop']
accuracy = np.array([0.28235294, 0.5372549, 0.39215686, 0.97254902, 0.61176471, 0.54901961, 0.31372549, 0.81176471,
                     0.76078431, 0.69803922])

print('\t{0:<10s}\t\t{1}'.format('Class', 'Accuracy'))
for i in range(len(labels)):
    # print('%s\t%-.4f' % ())
    print('\t{0:<10s}\t\t{1:>.4f}'.format(labels[i], accuracy[i]))

# print('ZHeHSIHJXABJKHD         asdjkdhaw')
# for i in range(len(labels)):
#     print('{0:>9s} {1:>}'.format(labels[i], labels[i]))
