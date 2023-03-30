import utils
import tasks
task_classes = [(tasks.DBIndexTask, 300),
                (tasks.RedisTask, 300), 
                (tasks.CardWikiTask, 300), 
                (tasks.LinnosTask, 300),
                (tasks.BloomCrimeTask, 500), 
                ]
utils.accuracy_comparison(task_classes)