import os


def data_download(data_dir):
    # ftp urls
    spring_s2 = "ftp://m1474000:m1474000@dataserv.ub.tum.de/ROIs1158_spring_s2.tar.gz"
    spring_lc = "ftp://m1474000:m1474000@dataserv.ub.tum.de/ROIs1158_spring_lc.tar.gz"
    summer_s2 = "ftp://m1474000:m1474000@dataserv.ub.tum.de/ROIs1868_summer_s2.tar.gz"
    summer_lc = "ftp://m1474000:m1474000@dataserv.ub.tum.de/ROIs1868_summer_lc.tar.gz"
    fall_s2 = "ftp://m1474000:m1474000@dataserv.ub.tum.de/ROIs1970_fall_s2.tar.gz"
    fall_lc = "ftp://m1474000:m1474000@dataserv.ub.tum.de/ROIs1970_fall_lc.tar.gz"
    winter_s2 = "ftp://m1474000:m1474000@dataserv.ub.tum.de/ROIs2017_winter_s2.tar.gz"
    winter_lc = "ftp://m1474000:m1474000@dataserv.ub.tum.de/ROIs2017_winter_lc.tar.gz"
    # create dir if does not exist
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    # download and extract files
    for url in [spring_s2, spring_lc, summer_s2, summer_lc,
                fall_s2, fall_lc, winter_s2, winter_lc]:
        fname = url.split('/')[-1]
        os.system("sudo wget {} -P {}".format(url, data_dir))
        os.system("sudo tar -xf {} -C {}".format(os.path.join(data_dir, fname), data_dir))
        os.system("sudo rm {}".format(os.path.join(data_dir, fname)))

