import time
import random
import netshare.ray as ray
from netshare import Generator

if __name__ == '__main__':
    # Change to False if you would not like to use Ray
    ray.config.enabled = False
    ray.init(address="auto")

    # configuration file
    generator = Generator(config="/home/aschoen/programs/NetShare/examples/netflow/config_xp2_netflow.json")

    # `work_folder` should not exist o/w an overwrite error will be thrown.
    # Please set the `worker_folder` as *absolute path*
    # if you are using Ray with multi-machine setup
    # since Ray has bugs when dealing with relative paths.
    t0 = time.time()
    generator.train(work_folder=f'/home/aschoen/my_storage/aschoen/xp2')
    t1 = time.time()
    dif1 = divmod(int(t1-t0),3600)
    print("Training time: {} hours,{} minutes and {} secounds".format(dif1[0], *divmod(dif1[1],60)))
    generator.generate(work_folder=f'/home/aschoen/my_storage/aschoen/xp2')
    t2 = time.time()
    dif2 = divmod(int(t2-t1),3600)
    print("Sampling time: {} hours,{} minutes and {} secounds".format(dif2[0], *divmod(dif2[1],60)))
    generator.visualize(work_folder=f'/home/aschoen/my_storage/aschoen/xp2')
    t3 = time.time()
    dif3 = divmod(int(t3 - t2),3600)
    print("Visualisation time! {} hors,{} minutes and {} secounds".format(dif3[0], *divmod(dif3[1],60)))

    ray.shutdown()
