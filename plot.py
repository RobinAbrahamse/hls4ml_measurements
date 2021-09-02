import numpy as np
import matplotlib.pyplot as plt

#ResNet50V2
i_tf_1p_u = 0.12597262845833332
i_tf_6p_u = 0.224228325625
i_tf_12p_u = 0.34375985010000004
i_tf_1_u = 0.090669487
i_tf_6_u = 0.218883674975
i_tf_12_u = 0.37557729605
v_tf_1p_u = 0.07945093358333333
v_tf_6p_u = 0.11616119635
v_tf_12p_u = 0.21168999075
v_tf_1_u = 0.09379796787083333
v_tf_6_u = 0.20113235165
v_tf_12_u = 0.34332302195
onnx_1_u = 0.011669655808333334
onnx_6_u = 0.056582695200000005
onnx_12_u = 0.12212727059999999
i_tf_1p_s = 0.02023369192976822
i_tf_6p_s = 0.0231470167643862
i_tf_12p_s = 0.03244391703880053
i_tf_1_s = 0.011030726435249381
i_tf_6_s = 0.019740917584809545
i_tf_12_s = 0.04788216100035207
v_tf_1p_s = 0.006164636652785351
v_tf_6p_s = 0.009766632205508727
v_tf_12p_s = 0.021385185174197255
v_tf_1_s = 0.013686987754138315
v_tf_6_s = 0.02156017181030051
v_tf_12_s = 0.031474551032870314
onnx_1_s = 0.004286203365708446
onnx_6_s = 0.012826749115171693
onnx_12_s = 0.029858114685223924

#MobileNetV2
i_tf_1p_u = 0.078833498325
i_tf_6p_u = 0.112146563625
i_tf_12p_u = 0.15672520269999998
i_tf_1_u = 0.0605978501625
i_tf_6_u = 0.12685228777500002
i_tf_12_u = 0.2010643525
v_tf_1p_u = 0.06298648822083333
v_tf_6p_u = 0.082033407475
v_tf_12p_u = 0.11627563355
v_tf_1_u = 0.058947731087
v_tf_6_u = 0.125148703825
v_tf_12_u = 0.1902109796
onnx_1_u = 0.005767888366666666
onnx_6_u = 0.033255509775
onnx_12_u = 0.07752292920000001
i_tf_1p_s = 0.008821896299718022
i_tf_6p_s = 0.016869984818418133
i_tf_12p_s = 0.017692724204359623
i_tf_1_s = 0.008471882360608892
i_tf_6_s = 0.01056438659704216
i_tf_12_s = 0.01984513321882651
v_tf_1p_s = 0.0035869764427269375
v_tf_6p_s = 0.009664954199849838
v_tf_12p_s = 0.010747685354501709
v_tf_1_s = 0.00973323849606068
v_tf_6_s = 0.013201554721923895
v_tf_12_s = 0.014638116661997418
onnx_1_s = 0.0013045214028653061
onnx_6_s = 0.006780127156718016
onnx_12_s = 0.013499418991872774

labels = ['intel-tf (.predict)', 'intel-tf', 'vanilla-tf (.predict)', 'vanilla-tf', 'ONNX']
x = np.arange(len(labels))
means_1 = [i_tf_1p_u, i_tf_1_u, v_tf_1p_u, v_tf_1_u, onnx_1_u]
means_6 = [i_tf_6p_u, i_tf_6_u, v_tf_6p_u, v_tf_6_u, onnx_6_u]
means_6 = [x/6 for x in means_6]
means_12 = [i_tf_12p_u, i_tf_12_u, v_tf_12p_u, v_tf_12_u, onnx_12_u]
means_12 = [x/12 for x in means_12]
stds_1 = [i_tf_1p_s, i_tf_1_s, v_tf_1p_s, v_tf_1_s, onnx_1_s]
stds_6 = [i_tf_6p_s, i_tf_6_s, v_tf_6p_s, v_tf_6_s, onnx_6_s]
stds_6 = [x/6 for x in stds_6]
stds_12 = [i_tf_12p_s, i_tf_12_s, v_tf_12p_s, v_tf_12_s, onnx_12_s]
stds_12 = [x/12 for x in stds_12]

w = 0.25
fig, ax = plt.subplots()
ax.bar(x - w, means_1, w, yerr=stds_1, align='center', alpha=0.5, label='Batch Size 1')
ax.bar(x, means_6, w, yerr=stds_6, align='center', alpha=0.5, label='Batch Size 6')
ax.bar(x + w, means_12, w, yerr=stds_12, align='center', alpha=0.5, label='Batch Size 12')
ax.set_ylabel('Inference latency per sample in s')
ax.set_title('MobileNetV2 Inference Latency')
#ax.set_title('ResNet50V2 Inference Latency')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.tight_layout()
plt.savefig('graph.png')
