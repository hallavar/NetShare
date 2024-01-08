import argparse
import random
import time
import netshare.ray as ray
from netshare import Generator

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog="NetShare",description="Generate NetFlow with NetShare")
    parser.add_argument("experience")
    
    args = parser.parse_args()

    xp = args.experience

    # Change to False if you would not like to use Ray
    ray.config.enabled = False
    ray.init(address="auto")

    # configuration file
    t0 = time.time()
    generator = Generator(config=f"/home/aschoen/programs/FlowChronicle/NetShare/examples/netflow/config_CIDDS_{xp}.json")
    t1 = time.time()
    dif1 = divmod(int(t1-t0),3600)
    print("Preprocessing time: {} hours, {} minutes, {} seconds".format(dif1[0], *divmod(dif1[1],60)))
    # `work_folder` should not exist o/w an overwrite error will be thrown.
    # Please set the `worker_folder` as *absolute path*
    # if you are using Ray with multi-machine setup
    # since Ray has bugs when dealing with relative paths.
    generator.train(work_folder=f'/home/aschoen/my_storage/aschoen/xp_CIDDS{xp[2]}')
    t2 = time.time()
    dif2 = divmod(int(t2-t1),3600)
    print("Training time: {} hours,{} minutes and {} secounds".format(dif2[0], *divmod(dif2[1],60)))
    generator.generate(work_folder=f'/home/aschoen/my_storage/aschoen/xp_CIDDS{xp[2]}')
    t3 = time.time()
    dif3 = divmod(int(t3-t2),3600)
    print("Sampling time: {} hours,{} minutes and {} secounds".format(dif3[0], *divmod(dif3[1],60)))
    generator.visualize(work_folder=f'/home/aschoen/my_storage/aschoen/xp_CIDDS{xp[2]}')
    

    ray.shutdown()
