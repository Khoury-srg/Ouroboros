
import utils
utils.cluster_bar_graph(
    [
    ("DBIndexTask", "LearnedIndex", 300), 
    ("RedisTask", "MemManager", 300), 
    ("CardWikiTask", "CardEsti", 300),
    ("LinnosTask", "LatPredictor", 300),  
    ("BloomCrimeTask", "BloomFilter", 1000),
    ],
    [
    ("00000-1","Normal"), 
    ("11000-1","Vanilla"), 
    ("11100-1","Spec. data"), 
    ("11110-1","Spec. data + EJ")
    ], 
    "../imgs/p2.png")

# utils.plot_acc_loss("../results/CardWikiTask11100-1_300.npz", "../imgs/cardesti_acc.pdf")
# utils.plot_acc_loss("../results/RedisTask11100-1_300.npz", "../imgs/redis_acc.pdf")
# utils.plot_acc_loss("../results/LinnosTask11100-1_300.npz", "../imgs/linnos_acc.pdf")
# utils.plot_acc_loss("../results/BloomCrimeTask11100-1_500.npz", "../imgs/bloom_acc.pdf")