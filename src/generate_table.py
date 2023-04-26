import utils
import tasks
task_classes = [
                (tasks.RedisTask, 300), 
                (tasks.CardWikiTask, 300), 
                (tasks.LinnosTask, 300),
                (tasks.BloomCrimeTask, 1000), 
                ]
utils.accuracy_comparison(task_classes)