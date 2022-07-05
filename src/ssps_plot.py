import matplotlib.pylab as plt

x = [2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100]

#v_ij data
#ssps1 = [8081849.895457074, 8839867.025279567, 9628559.98402493, 10372797.975169986, 11186303.281201825, 11997875.128566049, 12679797.485085478, 13180594.02929342, 13382505.45025703, 13323668.302036287]
#ssps2 = [8080359.400817183, 8796478.506856881, 9508903.09078534, 10150709.231338834, 10869045.455943994, 11592427.141108325, 12192085.836098872, 12698993.118963892, 13061316.731819998, 13225228.348801514]
#ssps3 = [8081217.250016224, 8562884.990404923, 8763243.556253606, 8788427.937814575, 8762901.154008511, 8632178.75098523, 8374224.917870592, 8027096.28223409, 7580028.341181385, 7057909.063211366]
#ssps4 = [8079748.900957672, 8716778.44549815, 9240211.433165973, 9634952.355695797, 9989173.860109264, 10241438.853641035, 10310333.160947595,  10212010.864996798,  9919467.432706773,  9431798.214965977]
#ssps5 = [8078948.779974811,  9030708.859319327,  10270835.175156357, 11577110.224443225, 13159936.829188595, 14940448.240091564, 16694025.164924055, 18384625.818761338, 19840983.26492571, 21026377.41823785]

# travel_demand_d
#ssps1 = [467010906.6974675, 510878579.20968807, 556514840.7806922, 599591165.5731665, 646658323.9232112,  693615226.8045955, 733094045.1511158, 762116876.0432943,  773883278.4741837, 770585684.8148849]
#ssps2 = [466917084.0475486,  508371931.26379204, 549615361.8225107,  586792600.9768529, 628368241.5687767,  670240486.9863567, 704967828.4020133, 734340473.3535854, 755362143.6624645, 764918818.9187225]
#ssps3 = [466969730.14811647, 494936292.7499776,  506685755.435477, 508314882.2465784,  506990185.16514015, 499561890.18873876,  484745418.9801853, 464741540.7834558, 438929910.9025523, 408753692.91077924]
#ssps4 = [466878639.3816075, 503782346.4650033, 534139430.9002401, 557072320.9472466, 577655641.117249, 592348276.2669351, 596446819.5267459,  590873017.5154899, 574037956.4116551, 545893971.6188245]
#ssps5 = [ 466829064.1769227, 521841978.1909386, 593463215.7528957, 668908169.7248865, 760293306.5038612,  863081622.0427299, 964324896.3625371, 1061943159.1362715,  1146062842.5618536, 1214557253.6636584]

# travel_demand_D_survey
#ssps1 = [10745135.322073828, 11753101.367202083, 12801859.38457762, 13791507.86244259, 14873261.977655055, 15952447.092660049, 16859239.717109077,  17525194.810434695, 17793721.713902414, 17715529.002026003]
#ssps2 = [10743153.432277255,  11695413.02591084, 12642759.300181495, 13496209.595962556, 14451417.650998952, 15413336.479235968,  16210743.742608309,  16884820.506682288,  17366644.51244524, 17584637.494434644]
#ssps3 = [10744293.981737275, 11384822.779403687, 11651312.033582259, 11684870.5448355,  11650989.326313458, 11477228.37299598, 11134291.859928006, 10672783.032282162, 10078387.854939224,  9384198.884143895]
#ssps4 = [10742342.305135895, 11589442.68128751, 12285500.022678968, 12810440.561643824, 13281498.521688167, 13616982.824022992, 13708639.455008602, 13577947.935789283, 13189011.935255531,  12540628.59574484]
#ssps5 = [10741278.60959407, 12006847.595934777, 13655846.029850163, 15392813.021887671, 17497514.25392872, 19865089.29179809,  22196866.53676039, 24444921.948500067, 26381514.094538394,  27957810.705824964]
# travel_demand_D_simulation
ssps1 = [9878269.789822347, 10804818.19817942, 11768865.66017599, 12678572.630835243, 13672950.298697622, 14664966.465876421, 15498501.237255879, 16110635.092915786, 16357422.648915784, 16285481.72998325]
ssps2 = [9876449.991127431, 10751782.841079745, 11622599.477701899, 12407092.12865287,  13285138.321097862, 14169350.1935069,  14902329.3793574, 15521932.563623372, 15964802.456043957, 16165142.965607453]
ssps3 = [9877497.939366838, 10466235.66192102,  10711108.01807687, 10741857.056244718, 10710623.850353988, 10550813.324098205, 10235497.769106196, 9811194.09723687, 9264743.629103255, 8626565.250041045]
ssps4 = [9875704.797248362, 10654358.191499218, 11294153.735302176, 11776640.887198016, 12209599.244500933, 12517930.613310227,  12602119.193846533,  12481914.678544536, 12124323.751814572, 11528238.55514311]
ssps5 = [9874727.740340259,  11038108.156604387,  12553989.44484482, 14150730.86566024,  16085529.643762833, 18261980.60272826, 20405511.371055733, 22472063.999539755, 24252280.965570047, 25701275.738977723]


l1 = plt.plot(x, ssps1, "mo-", label='ssps1')
l2 = plt.plot(x, ssps2, "ro-", label='ssps2')
l3 = plt.plot(x, ssps3, "bo-", label='ssps3')
l4 = plt.plot(x, ssps4, "yo-", label='ssps4')
l5 = plt.plot(x, ssps5, "go-", label='ssps5')
plt.xlabel('Year')
plt.xticks(ticks=x)

#plt.ylabel('Average distance weighted daily number of trips')
#plt.ylabel('Average distance weighted travel demand(straight-line distance)')
#plt.ylabel('Average distance weighted travel demand(travel survey based distance)')
plt.ylabel('Average distance weighted travel demand(simulation based distance)')

plt.legend()
plt.show()