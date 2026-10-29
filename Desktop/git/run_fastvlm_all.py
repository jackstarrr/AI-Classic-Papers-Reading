# -*- coding:gbk -*-
import os
import json
import argparse
import torch
import datetime
from PIL import Image
from fastvlm.llava.utils import disable_torch_init
from fastvlm.llava.conversation import conv_templates
from fastvlm.llava.model.builder import load_pretrained_model
from fastvlm.llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from fastvlm.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0')

PROMPTS_T0 = {
    '__name__': "PROMPTS_T0",
    'ALL': """
Please strictly judge whether the picture involves horror, porn, robbery, weapon, drug, fire, gambling, accidents,  or political,
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'.""",
    'horror': """
Please strictly judge whether the picture involves horror,
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'.""",
    'porn': """
Please strictly judge whether the picture involves porn,
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'.""",
    'robbery': """
Please strictly judge whether the picture involves robbery,
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'.""",
    'weapon': """
Please strictly judge whether the picture involves weapon,
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'.""",
    'drug': """
Please strictly judge whether the picture involves drug,
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'.""",
    'fire': """
Please strictly judge whether the picture involves fire,
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'""",
    'gambling': """
Please strictly judge whether the picture involves gambling,
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'.""",
    'accident': """
Please strictly judge whether the picture involves accident,
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'.""",
    'political': """
Please strictly judge whether the picture involves political,
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'."""
}

PROMPTS_T1_ROLE = {
    '__name__': "PROMPTS_T1_ROLE",
    'ALL': """
You're an expert in content detection. Please strictly judge whether the picture involves horror, porn, robbery, weapon, drug, fire, gambling, accidents,  or political,
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'.""",
    'horror': """
You're an expert in content detection. Please strictly judge whether the picture involves horror, 
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'.""",
    'porn': """
You're an expert in content detection. Please strictly judge whether the picture involves pornography, 
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'.""",
    'robbery': """
You're an expert in content detection. Please strictly judge whether the picture involves robbery behavior, 
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'.""",
    'weapon': """
You're an expert in content detection. Please strictly judge whether the picture involves weapon, 
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'.""",
    'drug': """
You're an expert in content detection. Please strictly judge whether the picture involves drug behavior, 
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'.""",
    'fire': """
You're an expert in content detection. Please strictly judge whether the picture involves fire or fire-related scenes,
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'""",
    'gambling': """
You're an expert in content detection. Please strictly judge whether the picture involves gambling, 
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'.""",
    'accident': """
You're an expert in content detection. Please strictly judge whether the picture involves an accident scene,
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'.""",
    'political': """
You're an expert in content detection. Please strictly judge whether the picture involves political, 
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'."""
}

PROMPTS_T2_ROLE_ELEMENTS = {
    '__name__': "PROMPTS_T2_ROLE_ELEMENTS",
    'ALL': """
You're an expert in content detection. Please strictly judge whether the picture involves horror, porn, robbery, weapon, drug, fire, gambling, accidents,  or political, the criteria are as follows:
    1. Horror: Corpse, skeleton, zombie skeleton, bone, bloodstain, ghost, spirit, zombie, zombie monster, bloody, limb deficiency, physiological lesion, corpse dissection, large-scale putrefaction, severed limbs and arms
	2. Porn: Naked and leaky genitals or sexual organs: male genitalia (penis, glans, testicles, semen, pubes), female genitalia (vulva, labia, clitoris, vagina, pubes), anus, buttocks and groin, female breast or nipples, or fluoroscopy/light and shadow of the above parts, Sexual behavior: sexual intercourse, masturbation, oral sex, fellatio, physical entanglement, etc, Sexual suggestion: erotic poses, teasing gestures, inducing sexual movements, Scene items: love hotel, adult film cover style costumes, sex props, restraint equipment, etc, Seduction signals: multi-dimensional superposition (e.g., nudity + provocative eyes + gestures), or stimulation designs for specific senses (e.g., wet lips close-up + ambiguous light and shadow)
	3. Robbery: Weapon use (sticks, knives, bricks, ironware, stones, bricks, glass bottles and other offensive tools).,Direct harm actions (punching, kicking, elbowing, scratching, fighting, kicking, beating, boxing, pushing, pulling, dragging, strangulation, throat_lock, head_beating, face_blow, violent_submission, armed_attack, stamping, etc., actions that cause physical pain), Control-based violence (strangling, choking, wrestling, throwing, etc.), Confrontational postures (shoving with hands, chest bumping, hair or clothing pulling, group confrontations in circle formations), Signs of emotional escalation (arm waving, shouting, etc.), Blood and injury outcomes (bleeding, bruises, lacerations, broken glasses, torn clothing), Environmental destruction clues (overturned tables/chairs, broken glass, bloodstains, scattered weapons)
	4. Weapon: Direct offensive weapons (firearms, knives, crossbows, explosives, tasers, tear gas, martial arts offensive equipment), Items that can be modified into offensive weapons (items with sharp/pointed parts such as chainsaws, axes, hammers, glass bottles, chairs, heavy objects, ropes with stones tied), Simulated and deterrent items (toy water guns, imitation knives)
	5. Drug: Poppy plant morphology, drug raw materials Poppy plant morphology, cannabis plant morphology Cannabis plant morphology, Drugs: powder, cannabis, cocaine, heroin, methamphetamine, drug powder, crystals, tablets, white piled substances, Drug paraphernalia: syringes, straws, pipes, packaging bags, bottled liquids, Drug use scenes: Drug use locations, illegal transaction sites, hands holding syringes/injectors, needles contacting skin for injection, mania, rolling eyes, fainting and other facial expressions
	6. Fire: visible fire, flames, fire tongues, sparks, embers, Smoke: thick smoke, black or white smoke, smoke columns rising, Burning objects: furniture, vehicles, buildings, forests, or waste piles on fire, Fire aftermath: charred remains, carbonized materials, melted metal, collapsed structures
	7. Gambling: playing cards, mahjong tiles, betting chips, gambling tokens, Gambling scenes: gaming tables, arcades, slot machines, roulette, gambling games, Gambling actions: drawing lots, dealing cards, drawing cards, lottery, rolling dice, roulette, placing bets
	8. Accidents: vehicle collisions, rollovers, rear-end crashes, scattered parts, skid marks, deployed airbags, Casualties: injured or unconscious persons, bloodstains, stretchers, medical personnel, onlookers, Infrastructure damage: broken bridges, collapsed buildings, fallen utility poles, sunken roads
	9. Political: Scenes of protests, demonstrations, riots, or rallies, Political figures, Political parties and their symbols/flags, Politically sensitive text or symbols, National symbols such as flags, Images of controversial political events
Judgment Principle:
    - Analyze the image content step by step to see if it contains any of the above
    - If more than one item is met at the same time, the highest severity level shall prevail
    - Please make a final judgment based on the above criteria
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'.""",
    'horror': """
You're an expert in content detection. Please strictly judge whether the picture involves horror, the criteria are as follows:
    1. Horror and suspense: Corpse, skeleton, zombie skeleton, bone, bloodstain, ghost, spirit, zombie, zombie monster, bloody, limb deficiency, physiological lesion, corpse dissection, large-scale putrefaction, severed limbs and arms
Judgment Principle:
    - Analyze the image content step by step to see if it contains any of the above
    - If more than one item is met at the same time, the highest severity level shall prevail
    - Please make a final judgment based on the above criteria
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'.""",
    'porn': """
You're an expert in content detection. Please strictly judge whether the picture involves pornography, the criteria are as follows:
    1. Naked and leaky genitals or sexual organs: male genitalia (penis, glans, testicles, semen, pubes), female genitalia (vulva, labia, clitoris, vagina, pubes), anus, buttocks and groin, female breast or nipples, or fluoroscopy/light and shadow of the above parts
    2. Sexual behavior: sexual intercourse, masturbation, oral sex, fellatio, physical entanglement, etc
    3. Sexual suggestion: erotic poses, teasing gestures, inducing sexual movements
    4. Scene items: love hotel, adult film cover style costumes, sex props, restraint equipment, etc
    5. Seduction signals: multi-dimensional superposition (e.g., nudity + provocative eyes + gestures), or stimulation designs for specific senses (e.g., wet lips close-up + ambiguous light and shadow)
Judgment Principle:
    - Analyze the image content step by step to see if it contains any of the above
    - If more than one item is met at the same time, the highest severity level shall prevail
    - Please make a final judgment based on the above criteria
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'.""",
    'robbery': """
You're an expert in content detection. Please strictly judge whether the picture involves robbery behavior, the criteria are as follows:
    1. Weapon use (sticks, knives, bricks, ironware, stones, bricks, glass bottles and other offensive tools)
    2. Direct harm actions (punching, kicking, elbowing, scratching, fighting, kicking, beating, boxing, pushing, pulling, dragging, strangulation, throat_lock, head_beating, face_blow, violent_submission, armed_attack, stamping, etc., actions that cause physical pain)
    3. Control-based violence (strangling, choking, wrestling, throwing, etc.)
    4. Confrontational postures (shoving with hands, chest bumping, hair or clothing pulling, group confrontations in circle formations)
    5. Signs of emotional escalation (arm waving, shouting, etc.)
    6. Blood and injury outcomes (bleeding, bruises, lacerations, broken glasses, torn clothing)
    7. Environmental destruction clues (overturned tables/chairs, broken glass, bloodstains, scattered weapons)
Judgment Principle:
    - Analyze the image content step by step to see if it contains any of the above
    - If more than one item is met at the same time, the highest severity level shall prevail
    - Please make a final judgment based on the above criteria
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'.""",
    'weapon': """
You're an expert in content detection. Please strictly judge whether the picture involves weapon, the criteria are as follows:
    1. Direct offensive weapons (firearms, knives, crossbows, explosives, tasers, tear gas, martial arts offensive equipment)
    2. Items that can be modified into offensive weapons (items with sharp/pointed parts such as chainsaws, axes, hammers, glass bottles, chairs, heavy objects, ropes with stones tied)
    3. Simulated and deterrent items (toy water guns, imitation knives)
Judgment Principle:
    - Analyze the image content step by step to see if it contains any of the above
    - If more than one item is met at the same time, the highest severity level shall prevail
    - Please make a final judgment based on the above criteria
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'.""",
    'drug': """
You're an expert in content detection. Please strictly judge whether the picture involves drug behavior, the criteria are as follows:
    1. Drug raw materials: Poppy plant morphology, drug raw materials Poppy plant morphology, cannabis plant morphology Cannabis plant morphology
    2. Drugs: powder, cannabis, cocaine, heroin, methamphetamine, drug powder, crystals, tablets, white piled substances
    3. Drug paraphernalia: syringes, straws, pipes, packaging bags, bottled liquids
    4. Drug use scenes: Drug use locations, illegal transaction sites, hands holding syringes/injectors, needles contacting skin for injection, mania, rolling eyes, fainting and other facial expressions
Judgment Principle:
    - Analyze the image content step by step to see if it contains any of the above
    - If more than one item is met at the same time, the highest severity level shall prevail
    - Please make a final judgment based on the above criteria
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'.""",
    'fire': """
You're an expert in content detection. Please strictly judge whether the picture involves fire or fire-related scenes. Criteria:
    1. Open flames: visible fire, flames, fire tongues, sparks, embers
    2. Smoke: thick smoke, black or white smoke, smoke columns rising
    3. Burning objects: furniture, vehicles, buildings, forests, or waste piles on fire
    4. Fire aftermath: charred remains, carbonized materials, melted metal, collapsed structures
Judgment Principle:
    - Analyze the image content step by step to see if it contains any of the above
    - If more than one item is met at the same time, the highest severity level shall prevail
    - Please make a final judgment based on the above criteria
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'""",
    'gambling': """
You're an expert in content detection. Please strictly judge whether the picture involves gambling, the criteria are as follows:
    1. Gambling tools: playing cards, mahjong tiles, betting chips, gambling tokens
    2. Gambling scenes: gaming tables, arcades, slot machines, roulette, gambling games
    3. Gambling actions: drawing lots, dealing cards, drawing cards, lottery, rolling dice, roulette, placing bets
Judgment Principle:
    - Analyze the image content step by step to see if it contains any of the above
    - If more than one item is met at the same time, the highest severity level shall prevail
    - Please make a final judgment based on the above criteria
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'.""",
    'accident': """
You're an expert in content detection. Please strictly judge whether the picture involves an accident scene. Criteria:
    1. Traffic accidents: vehicle collisions, rollovers, rear-end crashes, scattered parts, skid marks, deployed airbags
    2. Casualties: injured or unconscious persons, bloodstains, stretchers, medical personnel, onlookers
    3. Infrastructure damage: broken bridges, collapsed buildings, fallen utility poles, sunken roads
Judgment Principle:
    - Analyze the image content step by step to see if it contains any of the above
    - If more than one item is met at the same time, the highest severity level shall prevail
    - Please make a final judgment based on the above criteria
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'.""",
    'political': """
You're an expert in content detection. Please strictly judge whether the picture involves political, the criteria are as follows:
    1. political: Scenes of protests, demonstrations, riots, or rallies, Political figures, Political parties and their symbols/flags, Politically sensitive text or symbols, National symbols such as flags, Images of controversial political events
Judgment Principle:
    - Analyze the image content step by step to see if it contains any of the above
    - If more than one item is met at the same time, the highest severity level shall prevail
    - Please make a final judgment based on the above criteria
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'."""
}

PROMPTS_T4_ROLE_ELEMENTS_SAFETY = {
    '__name__': "PROMPTS_T4_ROLE_ELEMENTS_SAFETY",
    'ALL': """
You're an expert in content detection. Please strictly judge whether the picture involves horror, porn, robbery, weapon, drug, fire, gambling, accidents,  or political, the criteria are as follows:
    1. Horror: Corpse, skeleton, zombie skeleton, bone, bloodstain, ghost, spirit, zombie, zombie monster, bloody, limb deficiency, physiological lesion, corpse dissection, large-scale putrefaction, severed limbs and arms
	2. Porn: Naked and leaky genitals or sexual organs: male genitalia (penis, glans, testicles, semen, pubes), female genitalia (vulva, labia, clitoris, vagina, pubes), anus, buttocks and groin, female breast or nipples, or fluoroscopy/light and shadow of the above parts, Sexual behavior: sexual intercourse, masturbation, oral sex, fellatio, physical entanglement, etc, Sexual suggestion: erotic poses, teasing gestures, inducing sexual movements, Scene items: love hotel, adult film cover style costumes, sex props, restraint equipment, etc, Seduction signals: multi-dimensional superposition (e.g., nudity + provocative eyes + gestures), or stimulation designs for specific senses (e.g., wet lips close-up + ambiguous light and shadow)
	3. Robbery: Weapon use (sticks, knives, bricks, ironware, stones, bricks, glass bottles and other offensive tools).,Direct harm actions (punching, kicking, elbowing, scratching, fighting, kicking, beating, boxing, pushing, pulling, dragging, strangulation, throat_lock, head_beating, face_blow, violent_submission, armed_attack, stamping, etc., actions that cause physical pain), Control-based violence (strangling, choking, wrestling, throwing, etc.), Confrontational postures (shoving with hands, chest bumping, hair or clothing pulling, group confrontations in circle formations), Signs of emotional escalation (arm waving, shouting, etc.), Blood and injury outcomes (bleeding, bruises, lacerations, broken glasses, torn clothing), Environmental destruction clues (overturned tables/chairs, broken glass, bloodstains, scattered weapons)
	4. Weapon: Direct offensive weapons (firearms, knives, crossbows, explosives, tasers, tear gas, martial arts offensive equipment), Items that can be modified into offensive weapons (items with sharp/pointed parts such as chainsaws, axes, hammers, glass bottles, chairs, heavy objects, ropes with stones tied), Simulated and deterrent items (toy water guns, imitation knives)
	5. Drug: Poppy plant morphology, drug raw materials Poppy plant morphology, cannabis plant morphology Cannabis plant morphology, Drugs: powder, cannabis, cocaine, heroin, methamphetamine, drug powder, crystals, tablets, white piled substances, Drug paraphernalia: syringes, straws, pipes, packaging bags, bottled liquids, Drug use scenes: Drug use locations, illegal transaction sites, hands holding syringes/injectors, needles contacting skin for injection, mania, rolling eyes, fainting and other facial expressions
	6. Fire: visible fire, flames, fire tongues, sparks, embers, Smoke: thick smoke, black or white smoke, smoke columns rising, Burning objects: furniture, vehicles, buildings, forests, or waste piles on fire, Fire aftermath: charred remains, carbonized materials, melted metal, collapsed structures
	7. Gambling: playing cards, mahjong tiles, betting chips, gambling tokens, Gambling scenes: gaming tables, arcades, slot machines, roulette, gambling games, Gambling actions: drawing lots, dealing cards, drawing cards, lottery, rolling dice, roulette, placing bets
	8. Accidents: vehicle collisions, rollovers, rear-end crashes, scattered parts, skid marks, deployed airbags, Casualties: injured or unconscious persons, bloodstains, stretchers, medical personnel, onlookers, Infrastructure damage: broken bridges, collapsed buildings, fallen utility poles, sunken roads
	9. Political: Scenes of protests, demonstrations, riots, or rallies, Political figures, Political parties and their symbols/flags, Politically sensitive text or symbols, National symbols such as flags, Images of controversial political events
Judgment Principle:
    - Analyze the image content step by step to see if it contains any of the above
    - If more than one item is met at the same time, the highest severity level shall prevail
    - Output Better to misjudge than miss a judgment
    - In unclear or partial-view cases, prefer 'YES'
    - Please make a final judgment based on the above criteria
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'.""",
    'horror': """
You're an expert in content detection. Please strictly judge whether the picture involves horror, the criteria are as follows:
    1. Horror and suspense: Corpse, skeleton, zombie skeleton, bone, bloodstain, ghost, spirit, zombie, zombie monster, bloody, limb deficiency, physiological lesion, corpse dissection, large-scale putrefaction, severed limbs and arms
Judgment Principle:
    - Analyze the image content step by step to see if it contains any of the above
    - If more than one item is met at the same time, the highest severity level shall prevail
    - Output Better to misjudge than miss a judgment
    - In unclear or partial-view cases, prefer 'YES'
    - Please make a final judgment based on the above criteria
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'.""",
    'porn': """
You're an expert in content detection. Please strictly judge whether the picture involves pornography, the criteria are as follows:
    1. Naked and leaky genitals or sexual organs: male genitalia (penis, glans, testicles, semen, pubes), female genitalia (vulva, labia, clitoris, vagina, pubes), anus, buttocks and groin, female breast or nipples, or fluoroscopy/light and shadow of the above parts
    2. Sexual behavior: sexual intercourse, masturbation, oral sex, fellatio, physical entanglement, etc
    3. Sexual suggestion: erotic poses, teasing gestures, inducing sexual movements
    4. Scene items: love hotel, adult film cover style costumes, sex props, restraint equipment, etc
    5. Seduction signals: multi-dimensional superposition (e.g., nudity + provocative eyes + gestures), or stimulation designs for specific senses (e.g., wet lips close-up + ambiguous light and shadow)
Judgment Principle:
    - Analyze the image content step by step to see if it contains any of the above
    - If more than one item is met at the same time, the highest severity level shall prevail
    - Output Better to misjudge than miss a judgment
    - In unclear or partial-view cases, prefer 'YES'
    - Please make a final judgment based on the above criteria
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'.""",
    'robbery': """
You're an expert in content detection. Please strictly judge whether the picture involves robbery behavior, the criteria are as follows:
    1. Weapon use (sticks, knives, bricks, ironware, stones, bricks, glass bottles and other offensive tools)
    2. Direct harm actions (punching, kicking, elbowing, scratching, fighting, kicking, beating, boxing, pushing, pulling, dragging, strangulation, throat_lock, head_beating, face_blow, violent_submission, armed_attack, stamping, etc., actions that cause physical pain)
    3. Control-based violence (strangling, choking, wrestling, throwing, etc.)
    4. Confrontational postures (shoving with hands, chest bumping, hair or clothing pulling, group confrontations in circle formations)
    5. Signs of emotional escalation (arm waving, shouting, etc.)
    6. Blood and injury outcomes (bleeding, bruises, lacerations, broken glasses, torn clothing)
    7. Environmental destruction clues (overturned tables/chairs, broken glass, bloodstains, scattered weapons)
Judgment Principle:
    - Analyze the image content step by step to see if it contains any of the above
    - If more than one item is met at the same time, the highest severity level shall prevail
    - Output Better to misjudge than miss a judgment
    - In unclear or partial-view cases, prefer 'YES'
    - Please make a final judgment based on the above criteria
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'.""",
    'weapon': """
You're an expert in content detection. Please strictly judge whether the picture involves weapon, the criteria are as follows:
    1. Direct offensive weapons (firearms, knives, crossbows, explosives, tasers, tear gas, martial arts offensive equipment)
    2. Items that can be modified into offensive weapons (items with sharp/pointed parts such as chainsaws, axes, hammers, glass bottles, chairs, heavy objects, ropes with stones tied)
    3. Simulated and deterrent items (toy water guns, imitation knives)
Judgment Principle:
    - Analyze the image content step by step to see if it contains any of the above
    - If more than one item is met at the same time, the highest severity level shall prevail
    - Output Better to misjudge than miss a judgment
    - In unclear or partial-view cases, prefer 'YES'
    - Please make a final judgment based on the above criteria
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'.""",
    'drug': """
You're an expert in content detection. Please strictly judge whether the picture involves drug behavior, the criteria are as follows:
    1. Drug raw materials: Poppy plant morphology, drug raw materials Poppy plant morphology, cannabis plant morphology Cannabis plant morphology
    2. Drugs: powder, cannabis, cocaine, heroin, methamphetamine, drug powder, crystals, tablets, white piled substances
    3. Drug paraphernalia: syringes, straws, pipes, packaging bags, bottled liquids
    4. Drug use scenes: Drug use locations, illegal transaction sites, hands holding syringes/injectors, needles contacting skin for injection, mania, rolling eyes, fainting and other facial expressions
Judgment Principle:
    - Analyze the image content step by step to see if it contains any of the above
    - If more than one item is met at the same time, the highest severity level shall prevail
    - Output Better to misjudge than miss a judgment
    - In unclear or partial-view cases, prefer 'YES'
    - Please make a final judgment based on the above criteria
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'.""",
    'fire': """
You're an expert in content detection. Please strictly judge whether the picture involves fire or fire-related scenes. Criteria:
    1. Open flames: visible fire, flames, fire tongues, sparks, embers
    2. Smoke: thick smoke, black or white smoke, smoke columns rising
    3. Burning objects: furniture, vehicles, buildings, forests, or waste piles on fire
    4. Fire aftermath: charred remains, carbonized materials, melted metal, collapsed structures
Judgment Principle:
    - Analyze the image content step by step to see if it contains any of the above
    - If more than one item is met at the same time, the highest severity level shall prevail
    - Output Better to misjudge than miss a judgment
    - In unclear or partial-view cases, prefer 'YES'
    - Please make a final judgment based on the above criteria
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'""",
    'gambling': """
You're an expert in content detection. Please strictly judge whether the picture involves gambling, the criteria are as follows:
    1. Gambling tools: playing cards, mahjong tiles, betting chips, gambling tokens
    2. Gambling scenes: gaming tables, arcades, slot machines, roulette, gambling games
    3. Gambling actions: drawing lots, dealing cards, drawing cards, lottery, rolling dice, roulette, placing bets
Judgment Principle:
    - Analyze the image content step by step to see if it contains any of the above
    - If more than one item is met at the same time, the highest severity level shall prevail
    - Output Better to misjudge than miss a judgment
    - In unclear or partial-view cases, prefer 'YES'
    - Please make a final judgment based on the above criteria
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'.""",
    'accident': """
You're an expert in content detection. Please strictly judge whether the picture involves an accident scene. Criteria:
    1. Traffic accidents: vehicle collisions, rollovers, rear-end crashes, scattered parts, skid marks, deployed airbags
    2. Casualties: injured or unconscious persons, bloodstains, stretchers, medical personnel, onlookers
    3. Infrastructure damage: broken bridges, collapsed buildings, fallen utility poles, sunken roads
Judgment Principle:
    - Analyze the image content step by step to see if it contains any of the above
    - If more than one item is met at the same time, the highest severity level shall prevail
    - Output Better to misjudge than miss a judgment
    - In unclear or partial-view cases, prefer 'YES'
    - Please make a final judgment based on the above criteria
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'.""",
    'political': """
You're an expert in content detection. Please strictly judge whether the picture involves political, the criteria are as follows:
    1. political: Scenes of protests, demonstrations, riots, or rallies, Political figures, Political parties and their symbols/flags, Politically sensitive text or symbols, National symbols such as flags, Images of controversial political events
Judgment Principle:
    - Analyze the image content step by step to see if it contains any of the above
    - If more than one item is met at the same time, the highest severity level shall prevail
    - Output Better to misjudge than miss a judgment
    - In unclear or partial-view cases, prefer 'YES'
    - Please make a final judgment based on the above criteria
Output Requirements: Classification Task: After your analysis, your final answer must be on a new line and only contain the word 'YES' or 'NO'."""
}


# 文件夹下图片文件list获取
def generate_image_list(base_path, dataset_name):
    """
    遍历指定数据集文件夹，生成图像路径和标签的列表并保存到txt文件中
    Args:
        base_path (str): 数据集根目录路径
        outpath (str): 输出文件保存路径
        dataset_name (str): 数据集名称
    """
    dataset_path = os.path.join(base_path, dataset_name)

    # 检查数据集路径是否存在
    if not os.path.exists(dataset_path):
        print(f"警告: 数据集路径不存在 - {dataset_path}")
        return
    FILE_LIST = []

    # 遍历train, val, test三个文件夹
    # for split in ['train', 'val', 'test']:
    for split in ['test']:
        #split_path = dataset_path
        split_path = os.path.join(dataset_path, split)
        # 检查分割路径是否存在
        if not os.path.exists(split_path):
            print(f"警告: 分割路径不存在 - {split_path}")
            continue
        # 遍历该分割文件夹下的所有图片文件
        for image_file in os.listdir(split_path):
            image_path = os.path.join(split_path, image_file)
            # 确保是文件而不是文件夹
            if not os.path.isfile(image_path):
                continue
            # 检查是否是图片文件
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                # 所有图片的标签都为1
                label = '1'
                FILE_LIST.append(f"{image_path},{label}")
    return FILE_LIST


# 添加提示词
def convert_txt_to_json(file_path_list, dataset_name, prompts):
    """
    将包含图像路径和标签的txt文件转换为JSON格式，用于大模型推理
    Args:
    """
    prompt_data = []
    for line in file_path_list:
        line = line.strip()
        if not line:
            continue

        # 解析每行数据
        image_path, label = line.split(',')
        # 根据数据集名称获取对应的提示词
        # 正常数据需要走全部提示词

        prompt = prompts.get(dataset_name,
                             "Please analyze whether there is any anomaly in this image. If so, please describe the type of anomaly.")

        # 构建符合要求的JSON格式
        item = {
            "image_path": image_path,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        prompt_data.append(item)

    return prompt_data


def predict_single_image(model, tokenizer, image_processor, args, image_path, prompt):
    """
    对单张图像进行推理并返回结果
    """
    # 构建提示词
    qs = prompt
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt_final = conv.get_prompt()

    # 编码输入
    input_ids = tokenizer_image_token(prompt_final, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(
        torch.device(device))

    # 加载并预处理图像
    image = Image.open(image_path).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)[0]

    # 推理
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half(),
            image_sizes=[image.size],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=256,
            use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs


def run_inference_on_dataset(model, tokenizer, image_processor, args, prompt_data, output_final_file):
    """
    遍历所有数据并执行推理
    """
    results = []
    for item in prompt_data:
        image_path = item["image_path"]
        prompt = item["messages"][0]["content"]

        # 执行推理
        try:
            result = predict_single_image(model, tokenizer, image_processor, args, image_path, prompt)
            results.append({
                "image_path": image_path,
                "prompt": prompt,
                "response": result
            })
            # print(f"已处理: {image_path}")
        except Exception as e:
            print(f"处理失败 {image_path}: {e}")
            results.append({
                "image_path": image_path,
                "prompt": prompt,
                "response": f"错误: {str(e)}"
            })

        # 写入结果文件
    with open(output_final_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"已完成 {dataset} 数据集，结果保存至 {output_final_file}")


def predict(args, prompt_data, output_final_file):
    """
    主函数入口，加载模型并运行推理
    """
    # 移除生成配置以使用命令行参数
    model_path = os.path.expanduser(args.model_path)
    generation_config = None
    if os.path.exists(os.path.join(model_path, 'generation_config.json')):
        generation_config = os.path.join(model_path, '.generation_config.json')
        os.rename(os.path.join(model_path, 'generation_config.json'), generation_config)

    # 加载模型组件
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, device=device
    )

    # 设置pad token id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # 运行推理
    run_inference_on_dataset(model, tokenizer, image_processor, args, prompt_data, output_final_file)

    # 恢复生成配置
    if generation_config is not None:
        os.rename(generation_config, os.path.join(model_path, 'generation_config.json'))


def write_file_path(output_file_path, FILE_LIST):
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for i in FILE_LIST:
            f.write(i)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./fastvlm/model/llava-fastvithd_7b_stage2")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--conv_mode", type=str, default="qwen_2")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--base_data_path", type=str, default="/mnt/revocery_32T/Project/xujie/BTCD/reconstructed_dataset")
    parser.add_argument("--out_path", type=str, default="/home/mirror/BTCD/result_reconstructed_fastvlm")
    args = parser.parse_args()
    print(args)

    # datasets = ['normal', 'porn']
    # PROMPTS_LIST = [PROMPTS_T0]

    datasets = ['accident', 'drug', 'fire', 'gambling']
    #datasets = ['political', 'porn', 'horror', 'robbery', 'weapon']
    #datasets = ['political', 'porn', 'horror', 'robbery', 'weapon', 'accident', 'drug', 'fire', 'gambling']
    PROMPTS_LIST = [PROMPTS_T0, PROMPTS_T1_ROLE, PROMPTS_T2_ROLE_ELEMENTS, PROMPTS_T4_ROLE_ELEMENTS_SAFETY]

    # 为每个数据集生成图像列表
    for dataset in datasets:
        FILE_LIST = generate_image_list(args.base_data_path, dataset)
        # output_file_path = os.path.join(args.out_path, f"{dataset}_image_list.txt")
        # write_file_path(output_file_path, FILE_LIST)
        dataset_num = len(FILE_LIST)
        print(f"{dataset} 文件数量 {dataset_num}")
        for prompts in PROMPTS_LIST:
            if dataset == 'normal':
                for prompt_type in prompts:
                    if prompt_type == '__name__':
                        continue
                    prompt_data = convert_txt_to_json(FILE_LIST, prompt_type, prompts)
                    # # 写入JSON文件
                    # output_prompt_file = os.path.join(args.out_path, f"{dataset}_image_list.json")
                    # with open(output_prompt_file, 'w', encoding='utf-8') as f:
                    #     json.dump(prompt_data, f, ensure_ascii=False, indent=2)

                    tmp_model = args.model_path.split('/')[-1]
                    tmp_prompt = prompts['__name__']
                    output_final_file = os.path.join(args.out_path,
                                                     f"{dataset}_{prompt_type}_{tmp_model}_{tmp_prompt}_results.json")
                    predict(args, prompt_data, output_final_file)
            else:
                for prompt_type in [dataset]:
                #for prompt_type in [dataset, "ALL"]:
                    if prompt_type == '__name__':
                        continue
                    prompt_data = convert_txt_to_json(FILE_LIST, prompt_type, prompts)
                    # # 写入JSON文件
                    # output_prompt_file = os.path.join(args.out_path, f"{dataset}_image_list.json")
                    # with open(output_prompt_file, 'w', encoding='utf-8') as f:
                    #     json.dump(prompt_data, f, ensure_ascii=False, indent=2)

                    tmp_model = args.model_path.split('/')[-1]
                    tmp_prompt = prompts['__name__']
                    output_final_file = os.path.join(args.out_path,
                                                     f"{dataset}_{prompt_type}_{tmp_model}_{tmp_prompt}_results.json")
                    predict(args, prompt_data, output_final_file)
