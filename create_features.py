import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import sys
import pdb
from pdb import set_trace as st
import torch.nn.functional as F
import torchvision.transforms as T
from open_clip.clip import load, tokenize, _transform
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import pickle
from nltk.corpus import words
import nltk
import csv
import types
import argparse

parser = argparse.ArgumentParser(description='Training the Model')
parser.add_argument('--dataset', type=str, help='define dataset')
parser.add_argument('--patches', type=int, default=16, help="number of patches in each dimension")
args = parser.parse_args()

model, preprocess_val = load('ViT-L/14', jit=False)
model.eval()

class ImageDomainLabels(Dataset):
    def __init__(self, input_filename, transforms, root=None):
        with open(input_filename, 'r') as f:
            lines = f.readlines()
        filenames = [line.strip() for line in lines]
        self.images = [name.split(" ")[0] for name in filenames] 
        self.domains = [name.split(" ")[1] for name in filenames] 
        self.labels = [name.split(" ")[2] for name in filenames] 
        self.transforms = transforms
        self.root = root
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.root is not None:
            img_path = os.path.join(self.root, str(self.images[idx]))
        else:
            img_path = str(self.images[idx])
        images = self.transforms(Image.open(img_path))
        labels = self.labels[idx]
        domains = self.domains[idx]
        return images, labels, domains, img_path

def get_attention(model, images, block=5):
    # Attention maps list
    attention_maps = []

    # Define a new version of the attention method stores the attention weights
    def new_attention(self, x):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        attn_output, attn_weights = self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)  # Changed need_weights=True
        self.saved_attn_weights = attn_weights  # Store the attention weights in an instance variable
        return attn_output

    def attention_hook(module, input, output):
        # Extract the attention weights from the saved_attn_weights
        #original_shape = module.saved_attn_weights[:,0,1:].shape
        #new_shape = (original_shape[0], int(original_shape[1]**0.5), int(original_shape[1]**0.5))
        #attn_maps = module.saved_attn_weights[:, 0, 1:].reshape(*new_shape)
        
        #attention_maps.append(attn_maps)
        attention_maps.append(module.saved_attn_weights[:, 0, 1:])

    # Store the original attention method
    original_attention = model.visual.transformer.resblocks[block].attention
    # Replace default attention module of specific block with new attention module
    model.visual.transformer.resblocks[block].attention = types.MethodType(new_attention, model.visual.transformer.resblocks[block])
    # Register the forward hook to obtain attention maps
    hook_handle = model.visual.transformer.resblocks[block].register_forward_hook(attention_hook)

    # Make a forward pass
    _ = model.encode_image(images)

    # Restore the original attention method and unregister the hook
    model.visual.transformer.resblocks[block].attention = original_attention
    hook_handle.remove()

    return torch.cat(attention_maps, dim=0)

def get_features(model, images):
    # Features list
    features_list = []

    # Define a new version of the forward method stores all the tokens
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        if self.proj is not None:
            x = x @ self.proj
        return x

    def features_hook(module, input, output):
        # Extract the features
        features_list.append(output)

    # Store the original forward
    original_forward = model.visual.forward
    # Replace default forward module with new forward module to keep all the patches
    model.visual.forward= types.MethodType(forward, model.visual)

    # Register the forward hook to obtain features
    hook_handle = model.visual.register_forward_hook(features_hook)

    # Make a forward pass
    _ = model.encode_image(images)

    # Remove the hook after use to avoid storing features during other forward passes
    model.visual.forward = original_forward
    hook_handle.remove()
    return torch.cat(features_list, dim=0)

def save_dataset(dataloader, path_save):
    all_image_features, all_image_filenames, all_image_labels, all_image_domains = [], [], [], []
    with torch.no_grad():
        for images, labels, domains, filenames in dataloader:
            images = images.cuda(0, non_blocking=True)
            image_features = model.encode_image(images)           
            image_features = image_features / image_features.norm(dim=-1, keepdim=True) 
            all_image_features.append(image_features)
            for idx in range(len(filenames)):
                all_image_filenames.append(filenames[idx])
                all_image_labels.append(labels[idx])
                all_image_domains.append(domains[idx])
        all_image_features = torch.cat(all_image_features, dim=0)
        dict_save = {}
        dict_save['feats'] = all_image_features.data.cpu().numpy()
        dict_save['classes'] = all_image_labels
        dict_save['domains'] = all_image_domains
        dict_save['path'] = all_image_filenames
        with open(path_save,"wb") as f:
            pickle.dump(dict_save,f)

def save_DIY_dataset(dataloader, path_save, grid_size=16):
    import torchvision as tv
    all_image_features, all_image_filenames, all_image_labels, all_image_domains = [], [], [], []

    with torch.no_grad():
        counter = 0
        for images, labels, domains, filenames in dataloader:
            
            print(counter)
            start = time.time()
            for idx in range(images.shape[0]):
                image = images[idx]
                channels, height, width = image.size()
                patches = []
                patch_size = 224//grid_size
                for i in range(0, height, patch_size):
                    for j in range(0, width, patch_size):
                        patch = image[:, i:i+patch_size, j:j+patch_size]
                        patches.append(F.interpolate(patch.unsqueeze(0), size=224).squeeze())
                image_input = torch.stack(patches, dim=0).cuda(0, non_blocking=True)
                image_features = model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True) 
                all_image_features.append(image_features.cpu())
            for idx in range(len(filenames)):
                all_image_filenames.append(filenames[idx])
                all_image_labels.append(labels[idx])
                all_image_domains.append(domains[idx])
            print(time.time()-start)
            counter += images.shape[0]
        all_image_features = torch.stack(all_image_features, dim=0)
        dict_save = {}
        dict_save['feats'] = all_image_features.data.cpu().numpy()
        dict_save['classes'] = all_image_labels
        dict_save['domains'] = all_image_domains
        dict_save['path'] = all_image_filenames
        with open(path_save,"wb") as f:
            pickle.dump(dict_save,f)

def save_patch_dataset(dataloader, path_save):
    all_image_features, all_image_attn_maps_at_6, all_image_attn_maps_at_last, all_image_filenames, all_image_labels, all_image_domains = [], [], [], [], [], []
    with torch.no_grad():
        for images, labels, domains, filenames in dataloader:
            images = images.cuda(0, non_blocking=True)
            
            attn_maps_at_6 = get_attention(model, images)
            attn_maps_at_last = get_attention(model, images, block=-1)

            all_features = get_features(model, images)       
            all_features = all_features / all_features.norm(dim=-1, keepdim=True) 
            
            all_image_features.append(all_features.cpu())
            all_image_attn_maps_at_6.append(attn_maps_at_6.cpu())
            all_image_attn_maps_at_last.append(attn_maps_at_last.cpu())

            for idx in range(len(filenames)):
                all_image_filenames.append(filenames[idx])
                all_image_labels.append(labels[idx])
                all_image_domains.append(domains[idx])
        
        all_image_features = torch.cat(all_image_features, dim=0)
        all_image_attn_maps_at_6 = torch.cat(all_image_attn_maps_at_6, dim=0)
        all_image_attn_maps_at_last = torch.cat(all_image_attn_maps_at_last, dim=0)
        
        dict_save = {}
        dict_save['cls'] = all_image_features[:,0,:].data.cpu().numpy()
        dict_save['patches'] = all_image_features[:,1:,:].data.cpu().numpy()
        dict_save['attn@6'] = all_image_attn_maps_at_6.data.cpu().numpy()
        dict_save['attn@24'] = all_image_attn_maps_at_last.data.cpu().numpy()
        dict_save['classes'] = all_image_labels
        dict_save['domains'] = all_image_domains
        dict_save['path'] = all_image_filenames

        with open(path_save,"wb") as f:
            pickle.dump(dict_save,f)

modality = 'image'
mode = 'pacs'
save_patch = False
diy = False
grid_size = vars(args)['patches']

if vars(args)['dataset'] is not None:
    if vars(args)['dataset'].lower() in ['in', 'imgnet', 'imagenet', 'imagenet_r', 'imagenet-r', 'imagenetr']:
        mode = 'imagenet_r'
    elif vars(args)['dataset'].lower() in ['nico', 'nico++']:
        mode = 'nico'
    elif vars(args)['dataset'].lower() in ['pacs']:
        mode = 'pacs'
    elif vars(args)['dataset'].lower() in ['dn', 'minidn', 'domainnet', 'minidomainnet']:
        mode = 'minidn'
    elif vars(args)['dataset'].lower() in ['lueven']:
        mode = 'lueven'
    elif vars(args)['dataset'].lower() in ['tokyo', 'tokyo247']:
        mode = 'tokyo247'

if modality == 'corpus':
    if mode == 'imagenet':
        names = ["kit_fox","English_setter","Siberian_husky","Australian_terrier","English_springer","grey_whale","lesser_panda","Egyptian_cat","ibex","Persian_cat","cougar","gazelle","porcupine","sea_lion","malamute","badger","Great_Dane","Walker_hound","Welsh_springer_spaniel","whippet","Scottish_deerhound","killer_whale","mink","African_elephant","Weimaraner","soft-coated_wheaten_terrier","Dandie_Dinmont","red_wolf","Old_English_sheepdog","jaguar","otterhound","bloodhound","Airedale","hyena","meerkat","giant_schnauzer","titi","three-toed_sloth","sorrel","black-footed_ferret","dalmatian","black-and-tan_coonhound","papillon","skunk","Staffordshire_bullterrier","Mexican_hairless","Bouvier_des_Flandres","weasel","miniature_poodle","Cardigan","malinois","bighorn","fox_squirrel","colobus","tiger_cat","Lhasa","impala","coyote","Yorkshire_terrier","Newfoundland","brown_bear","red_fox","Norwegian_elkhound","Rottweiler","hartebeest","Saluki","grey_fox","schipperke","Pekinese","Brabancon_griffon","West_Highland_white_terrier","Sealyham_terrier","guenon","mongoose","indri","tiger","Irish_wolfhound","wild_boar","EntleBucher","zebra","ram","French_bulldog","orangutan","basenji","leopard","Bernese_mountain_dog","Maltese_dog","Norfolk_terrier","toy_terrier","vizsla","cairn","squirrel_monkey","groenendael","clumber","Siamese_cat","chimpanzee","komondor","Afghan_hound","Japanese_spaniel","proboscis_monkey","guinea_pig","white_wolf","ice_bear","gorilla","borzoi","toy_poodle","Kerry_blue_terrier","ox","Scotch_terrier","Tibetan_mastiff","spider_monkey","Doberman","Boston_bull","Greater_Swiss_Mountain_dog","Appenzeller","Shih-Tzu","Irish_water_spaniel","Pomeranian","Bedlington_terrier","warthog","Arabian_camel","siamang","miniature_schnauzer","collie","golden_retriever","Irish_terrier","affenpinscher","Border_collie","hare","boxer","silky_terrier","beagle","Leonberg","German_short-haired_pointer","patas","dhole","baboon","macaque","Chesapeake_Bay_retriever","bull_mastiff","kuvasz","capuchin","pug","curly-coated_retriever","Norwich_terrier","flat-coated_retriever","hog","keeshond","Eskimo_dog","Brittany_spaniel","standard_poodle","Lakeland_terrier","snow_leopard","Gordon_setter","dingo","standard_schnauzer","hamster","Tibetan_terrier","Arctic_fox","wire-haired_fox_terrier","basset","water_buffalo","American_black_bear","Angora","bison","howler_monkey","hippopotamus","chow","giant_panda","American_Staffordshire_terrier","Shetland_sheepdog","Great_Pyrenees","Chihuahua","tabby","marmoset","Labrador_retriever","Saint_Bernard","armadillo","Samoyed","bluetick","redbone","polecat","marmot","kelpie","gibbon","llama","miniature_pinscher","wood_rabbit","Italian_greyhound","lion","cocker_spaniel","Irish_setter","dugong","Indian_elephant","beaver","Sussex_spaniel","Pembroke","Blenheim_spaniel","Madagascar_cat","Rhodesian_ridgeback","lynx","African_hunting_dog","langur","Ibizan_hound","timber_wolf","cheetah","English_foxhound","briard","sloth_bear","Border_terrier","German_shepherd","otter","koala","tusker","echidna","wallaby","platypus","wombat","revolver","umbrella","schooner","soccer_ball","accordion","ant","starfish","chambered_nautilus","grand_piano","laptop","strawberry","airliner","warplane","airship","balloon","space_shuttle","fireboat","gondola","speedboat","lifeboat","canoe","yawl","catamaran","trimaran","container_ship","liner","pirate","aircraft_carrier","submarine","wreck","half_track","tank","missile","bobsled","dogsled","bicycle-built-for-two","mountain_bike","freight_car","passenger_car","barrow","shopping_cart","motor_scooter","forklift","electric_locomotive","steam_locomotive","amphibian","ambulance","beach_wagon","cab","convertible","jeep","limousine","minivan","Model_T","racer","sports_car","go-kart","golfcart","moped","snowplow","fire_engine","garbage_truck","pickup","tow_truck","trailer_truck","moving_van","police_van","recreational_vehicle","streetcar","snowmobile","tractor","mobile_home","tricycle","unicycle","horse_cart","jinrikisha","oxcart","bassinet","cradle","crib","four-poster","bookcase","china_cabinet","medicine_chest","chiffonier","table_lamp","file","park_bench","barber_chair","throne","folding_chair","rocking_chair","studio_couch","toilet_seat","desk","pool_table","dining_table","entertainment_center","wardrobe","Granny_Smith","orange","lemon","fig","pineapple","banana","jackfruit","custard_apple","pomegranate","acorn","hip","ear","rapeseed","corn","buckeye","organ","upright","chime","drum","gong","maraca","marimba","steel_drum","banjo","cello","violin","harp","acoustic_guitar","electric_guitar","cornet","French_horn","trombone","harmonica","ocarina","panpipe","bassoon","oboe","sax","flute","daisy","yellow_lady's_slipper","cliff","valley","alp","volcano","promontory","sandbar","coral_reef","lakeside","seashore","geyser","hatchet","cleaver","letter_opener","plane","power_drill","lawn_mower","hammer","corkscrew","can_opener","plunger","screwdriver","shovel","plow","chain_saw","cock","hen","ostrich","brambling","goldfinch","house_finch","junco","indigo_bunting","robin","bulbul","jay","magpie","chickadee","water_ouzel","kite","bald_eagle","vulture","great_grey_owl","black_grouse","ptarmigan","ruffed_grouse","prairie_chicken","peacock","quail","partridge","African_grey","macaw","sulphur-crested_cockatoo","lorikeet","coucal","bee_eater","hornbill","hummingbird","jacamar","toucan","drake","red-breasted_merganser","goose","black_swan","white_stork","black_stork","spoonbill","flamingo","American_egret","little_blue_heron","bittern","crane","limpkin","American_coot","bustard","ruddy_turnstone","red-backed_sandpiper","redshank","dowitcher","oystercatcher","European_gallinule","pelican","king_penguin","albatross","great_white_shark","tiger_shark","hammerhead","electric_ray","stingray","barracouta","coho","tench","goldfish","eel","rock_beauty","anemone_fish","lionfish","puffer","sturgeon","gar","loggerhead","leatherback_turtle","mud_turtle","terrapin","box_turtle","banded_gecko","common_iguana","American_chameleon","whiptail","agama","frilled_lizard","alligator_lizard","Gila_monster","green_lizard","African_chameleon","Komodo_dragon","triceratops","African_crocodile","American_alligator","thunder_snake","ringneck_snake","hognose_snake","green_snake","king_snake","garter_snake","water_snake","vine_snake","night_snake","boa_constrictor","rock_python","Indian_cobra","green_mamba","sea_snake","horned_viper","diamondback","sidewinder","European_fire_salamander","common_newt","eft","spotted_salamander","axolotl","bullfrog","tree_frog","tailed_frog","whistle","wing","paintbrush","hand_blower","oxygen_mask","snorkel","loudspeaker","microphone","screen","mouse","electric_fan","oil_filter","strainer","space_heater","stove","guillotine","barometer","rule","odometer","scale","analog_clock","digital_clock","wall_clock","hourglass","sundial","parking_meter","stopwatch","digital_watch","stethoscope","syringe","magnetic_compass","binoculars","projector","sunglasses","loupe","radio_telescope","bow","cannon","assault_rifle","rifle","projectile","computer_keyboard","typewriter_keyboard","crane","lighter","abacus","cash_machine","slide_rule","desktop_computer","hand-held_computer","notebook","web_site","harvester","thresher","printer","slot","vending_machine","sewing_machine","joystick","switch","hook","car_wheel","paddlewheel","pinwheel","potter's_wheel","gas_pump","carousel","swing","reel","radiator","puck","hard_disc","sunglass","pick","car_mirror","solar_dish","remote_control","disk_brake","buckle","hair_slide","knot","combination_lock","padlock","nail","safety_pin","screw","muzzle","seat_belt","ski","candle","jack-o'-lantern","spotlight","torch","neck_brace","pier","tripod","maypole","mousetrap","spider_web","trilobite","harvestman","scorpion","black_and_gold_garden_spider","barn_spider","garden_spider","black_widow","tarantula","wolf_spider","tick","centipede","isopod","Dungeness_crab","rock_crab","fiddler_crab","king_crab","American_lobster","spiny_lobster","crayfish","hermit_crab","tiger_beetle","ladybug","ground_beetle","long-horned_beetle","leaf_beetle","dung_beetle","rhinoceros_beetle","weevil","fly","bee","grasshopper","cricket","walking_stick","cockroach","mantis","cicada","leafhopper","lacewing","dragonfly","damselfly","admiral","ringlet","monarch","cabbage_butterfly","sulphur_butterfly","lycaenid","jellyfish","sea_anemone","brain_coral","flatworm","nematode","conch","snail","slug","sea_slug","chiton","sea_urchin","sea_cucumber","iron","espresso_maker","microwave","Dutch_oven","rotisserie","toaster","waffle_iron","vacuum","dishwasher","refrigerator","washer","Crock_Pot","frying_pan","wok","caldron","coffeepot","teapot","spatula","altar","triumphal_arch","patio","steel_arch_bridge","suspension_bridge","viaduct","barn","greenhouse","palace","monastery","library","apiary","boathouse","church","mosque","stupa","planetarium","restaurant","cinema","home_theater","lumbermill","coil","obelisk","totem_pole","castle","prison","grocery_store","bakery","barbershop","bookshop","butcher_shop","confectionery","shoe_shop","tobacco_shop","toyshop","fountain","cliff_dwelling","yurt","dock","brass","megalith","bannister","breakwater","dam","chainlink_fence","picket_fence","worm_fence","stone_wall","grille","sliding_door","turnstile","mountain_tent","scoreboard","honeycomb","plate_rack","pedestal","beacon","mashed_potato","bell_pepper","head_cabbage","broccoli","cauliflower","zucchini","spaghetti_squash","acorn_squash","butternut_squash","cucumber","artichoke","cardoon","mushroom","shower_curtain","jean","carton","handkerchief","sandal","ashcan","safe","plate","necklace","croquet_ball","fur_coat","thimble","pajama","running_shoe","cocktail_shaker","chest","manhole_cover","modem","tub","tray","balance_beam","bagel","prayer_rug","kimono","hot_pot","whiskey_jug","knee_pad","book_jacket","spindle","ski_mask","beer_bottle","crash_helmet","bottlecap","tile_roof","mask","maillot","Petri_dish","football_helmet","bathing_cap","teddy","holster","pop_bottle","photocopier","vestment","crossword_puzzle","golf_ball","trifle","suit","water_tower","feather_boa","cloak","red_wine","drumstick","shield","Christmas_stocking","hoopskirt","menu","stage","bonnet","meat_loaf","baseball","face_powder","scabbard","sunscreen","beer_glass","hen-of-the-woods","guacamole","lampshade","wool","hay","bow_tie","mailbag","water_jug","bucket","dishrag","soup_bowl","eggnog","mortar","trench_coat","paddle","chain","swab","mixing_bowl","potpie","wine_bottle","shoji","bulletproof_vest","drilling_platform","binder","cardigan","sweatshirt","pot","birdhouse","hamper","ping-pong_ball","pencil_box","pay-phone","consomme","apron","punching_bag","backpack","groom","bearskin","pencil_sharpener","broom","mosquito_net","abaya","mortarboard","poncho","crutch","Polaroid_camera","space_bar","cup","racket","traffic_light","quill","radio","dough","cuirass","military_uniform","lipstick","shower_cap","monitor","oscilloscope","mitten","brassiere","French_loaf","vase","milk_can","rugby_ball","paper_towel","earthstar","envelope","miniskirt","cowboy_hat","trolleybus","perfume","bathtub","hotdog","coral_fungus","bullet_train","pillow","toilet_tissue","cassette","carpenter's_kit","ladle","stinkhorn","lotion","hair_spray","academic_gown","dome","crate","wig","burrito","pill_bottle","chain_mail","theater_curtain","window_shade","barrel","washbasin","ballpoint","basketball","bath_towel","cowboy_boot","gown","window_screen","agaric","cellular_telephone","nipple","barbell","mailbox","lab_coat","fire_screen","minibus","packet","maze","pole","horizontal_bar","sombrero","pickelhaube","rain_barrel","wallet","cassette_player","comic_book","piggy_bank","street_sign","bell_cote","fountain_pen","Windsor_tie","volleyball","overskirt","sarong","purse","bolo_tie","bib","parachute","sleeping_bag","television","swimming_trunks","measuring_cup","espresso","pizza","breastplate","shopping_basket","wooden_spoon","saltshaker","chocolate_sauce","ballplayer","goblet","gyromitra","stretcher","water_bottle","dial_telephone","soap_dispenser","jersey","school_bus","jigsaw_puzzle","plastic_bag","reflex_camera","diaper","Band_Aid","ice_lolly","velvet","tennis_ball","gasmask","doormat","Loafer","ice_cream","pretzel","quilt","maillot","tape_player","clog","iPod","bolete","scuba_diver","pitcher","matchstick","bikini","sock","CD_player","lens_cap","thatch","vault","beaker","bubble","cheeseburger","parallel_bars","flagpole","coffee_mug","rubber_eraser","stole","carbonara","dumbbell"]
        save_dir = "./clip_features/corpus/imagenet_names.pkl"
    elif mode == 'nltk_words':
        nltk.download('words')
        names = words.words()
        save_dir = "./clip_features/corpus/nltk_words.pkl"
    elif mode == 'open_images':
        names = []
        with open('./clip_features/corpus/open_image_v7_class_names.csv', newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                names.append(row[0])
        save_dir = "./clip_features/corpus/open_image_v7_class_names.pkl"

    all_text_features = []
    all_text = []
    with torch.no_grad():
        for idx, actual_text in enumerate(names):
            print(idx, end='\r')

            text = []
            text_tokens = tokenize(actual_text)
            text.append(text_tokens)

            text = torch.cat(text, dim=0)
            text = text.cuda(0, non_blocking=True)
            text_feature = model.encode_text(text)
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)

            all_text_features.append(text_feature)
            all_text.append(actual_text)
        all_text_features = torch.cat(all_text_features, dim=0)
        dict_save = {}
        dict_save['feats'] = all_text_features.data.cpu().numpy()
        dict_save['prompts'] = all_text
        with open(save_dir,"wb") as f:
            pickle.dump(dict_save,f)
elif modality == 'image':
    
    if mode == 'imagenet_r':
        full_dataset_path = "./data/imgnet/all_files.csv"
        full_dataset = ImageDomainLabels(full_dataset_path, root='./data', transforms=preprocess_val)
        full_dataloader = DataLoader(full_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
        if diy:
            save_DIY_dataset(full_dataloader, './clip_features/imagenet_r/full_imgnet_diy_patch_'+str(grid_size)+'_features.pkl', grid_size=grid_size)
        elif save_patch:
            save_patch_dataset(full_dataloader, './clip_features/imagenet_r/full_imgnet_patch_features.pkl')
        else:
            save_dataset(full_dataloader, './clip_features/imagenet_r/full_imgnet_features2.pkl')
    elif mode == 'nico':
        full_dataset_path = "./data/nico/all_files.csv"
        database_path = "./data/nico/database_files.csv"
        query_path = "./data/nico/query_files.csv"
        full_dataset = ImageDomainLabels(full_dataset_path, root='./data', transforms=preprocess_val)
        database_dataset = ImageDomainLabels(database_path, root='./data', transforms=preprocess_val)
        query_dataset = ImageDomainLabels(query_path, root='./data', transforms=preprocess_val)
        full_dataloader = DataLoader(full_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
        database_dataloader = DataLoader(database_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
        query_dataloader = DataLoader(query_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
        if diy:
            #save_DIY_dataset(full_dataloader, './clip_features/nico/full_nico_diy_patch_'+str(grid_size)+'_features.pkl', grid_size=grid_size)
            save_DIY_dataset(database_dataloader, './clip_features/nico/database_nico_diy_patch_'+str(grid_size)+'_features.pkl', grid_size=grid_size)
            save_DIY_dataset(query_dataloader, './clip_features/nico/query_nico_diy_patch_'+str(grid_size)+'_features.pkl', grid_size=grid_size)
        elif save_patch:
            #save_patch_dataset(full_dataloader, './clip_features/nico/full_nico_patch_features.pkl')
            save_patch_dataset(database_dataloader, './clip_features/nico/database_nico_patch_features.pkl')
            save_patch_dataset(query_dataloader, './clip_features/nico/query_nico_patch_features.pkl')
        else:
            save_dataset(full_dataloader, './clip_features/nico/full_nico_features.pkl')
            save_dataset(database_dataloader, './clip_features/nico/database_nico_features.pkl')
            save_dataset(query_dataloader, './clip_features/nico/query_nico_features.pkl')
    elif mode == 'minidn':
        full_dataset_path = "./data/minidn/all_files.csv"
        database_path = "./data/minidn/database_files.csv"
        query_path = "./data/minidn/query_files.csv"
        full_dataset = ImageDomainLabels(full_dataset_path, root='./data', transforms=preprocess_val)
        database_dataset = ImageDomainLabels(database_path, root='./data', transforms=preprocess_val)
        query_dataset = ImageDomainLabels(query_path, root='./data', transforms=preprocess_val)
        full_dataloader = DataLoader(full_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
        database_dataloader = DataLoader(database_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
        query_dataloader = DataLoader(query_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
        if diy:
            save_DIY_dataset(full_dataloader, './clip_features/minidn/full_minidn_diy_patch_'+str(grid_size)+'_features.pkl', grid_size=grid_size)
            save_DIY_dataset(database_dataloader, './clip_features/minidn/database_minidn_diy_patch_'+str(grid_size)+'_features.pkl', grid_size=grid_size)
            save_DIY_dataset(query_dataloader, './clip_features/minidn/query_minidn_diy_patch_'+str(grid_size)+'_features.pkl', grid_size=grid_size)
        elif save_patch:
            save_patch_dataset(full_dataloader, './clip_features/minidn/full_minidn_patch_features.pkl')
            save_patch_dataset(database_dataloader, './clip_features/minidn/database_minidn_patch_features.pkl')
            save_patch_dataset(query_dataloader, './clip_features/minidn/query_minidn_patch_features.pkl')
        else:
            save_dataset(full_dataloader, './clip_features/minidn/full_minidn_features.pkl')
            save_dataset(database_dataloader, './clip_features/minidn/database_minidn_features.pkl')
            save_dataset(query_dataloader, './clip_features/minidn/query_minidn_features.pkl')
    elif mode == 'lueven':
        full_dataset_path = "./data/lueven/all_files.csv"
        full_dataset = ImageDomainLabels(full_dataset_path, root='./data', transforms=preprocess_val)
        full_dataloader = DataLoader(full_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
        if diy:
            save_DIY_dataset(full_dataloader, './clip_features/lueven/full_lueven_diy_patch_'+str(grid_size)+'_features.pkl', grid_size=grid_size)
        elif save_patch:
            save_patch_dataset(full_dataloader, './clip_features/lueven/full_lueven_patch_features.pkl')
        else:
            save_dataset(full_dataloader, './clip_features/lueven/full_lueven_features.pkl')
    elif mode == 'pacs':
        full_dataset_path = "./data/PACS/all_files.csv"
        database_path = "./data/PACS/database_files.csv"
        query_path = "./data/PACS/query_files.csv"
        full_dataset = ImageDomainLabels(full_dataset_path, root='./data', transforms=preprocess_val)
        database_dataset = ImageDomainLabels(database_path, root='./data', transforms=preprocess_val)
        query_dataset = ImageDomainLabels(query_path, root='./data', transforms=preprocess_val)
        full_dataloader = DataLoader(full_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
        database_dataloader = DataLoader(database_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
        query_dataloader = DataLoader(query_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
        if diy:
            save_DIY_dataset(full_dataloader, './clip_features/PACS/full_pacs_diy_patch_'+str(grid_size)+'_features.pkl', grid_size=grid_size)
            save_DIY_dataset(database_dataloader, './clip_features/PACS/database_pacs_diy_patch_'+str(grid_size)+'_features.pkl', grid_size=grid_size)
            save_DIY_dataset(query_dataloader, './clip_features/PACS/query_pacs_diy_patch_'+str(grid_size)+'_features.pkl', grid_size=grid_size)
        elif save_patch:
            save_patch_dataset(full_dataloader, './clip_features/PACS/full_pacs_patch_features.pkl')
            save_patch_dataset(database_dataloader, './clip_features/PACS/database_pacs_patch_features.pkl')
            save_patch_dataset(query_dataloader, './clip_features/PACS/query_pacs_patch_features.pkl')
        else:
            save_dataset(full_dataloader, './clip_features/PACS/full_pacs_features.pkl')
            save_dataset(database_dataloader, './clip_features/PACS/database_pacs_features.pkl')
            save_dataset(query_dataloader, './clip_features/PACS/query_pacs_features.pkl')
    elif mode == 'tokyo247':
        full_dataset_path = "./data/Tokyo247/all_files.csv"
        full_dataset = ImageDomainLabels(full_dataset_path, root='./data', transforms=preprocess_val)
        full_dataloader = DataLoader(full_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
        if diy:
            save_DIY_dataset(full_dataloader, './clip_features/tokyo247/full_tokyo247_diy_patch_'+str(grid_size)+'_features.pkl', grid_size=grid_size)
        elif save_patch:
            save_patch_dataset(full_dataloader, './clip_features/tokyo247/full_tokyo247_patch_features.pkl')
        else:
            save_dataset(full_dataloader, './clip_features/tokyo247/full_tokyo247_features.pkl')
    