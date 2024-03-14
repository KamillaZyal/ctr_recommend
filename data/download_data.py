import hydra
import urllib.request

@hydra.main(config_path="../configs", config_name="config",version_base="1.3")
def main(cfg):
    for link,save_path in [(cfg.data.train_data_link,cfg.data.train_data_path),(cfg.data.test_data_link,cfg.data.test_data_path)]:
        logo = urllib.request.urlopen(link).read()
        f = open(save_path, "wb")
        f.write(logo)
        f.close()

if __name__ == "__main__":
    main()