'''Project specific constants and configuration. Non-HuggingFace 
defaults for various model parameters'''

import os
import torch

######################################################################
# Perplexity ratio scoring globals ###################################
######################################################################

# Get path to this config file so that we can define
# other paths relative to it
PROJECT_ROOT_PATH=os.path.dirname(os.path.realpath(__file__))

# Data paths
RAW_DATA_PATH=f'{PROJECT_ROOT_PATH}/data/raw_data'
INTERMEDIATE_DATA_PATH=f'{PROJECT_ROOT_PATH}/data/intermediate_data'

# Logs path
LOG_PATH=f'{PROJECT_ROOT_PATH}/logs'

# Path to save notebook plots
PLOT_PATH=f'{PROJECT_ROOT_PATH}/notebooks/plots'

######################################################################
# Old v1.1.0 classifier stuff, left in temporarily for compatibility #
######################################################################

# Other project paths
DATA_PATH=f'{PROJECT_ROOT_PATH}/data'
BENCHMARKING_DATA_PATH=f'{DATA_PATH}/benchmarking'
HANS_DATA_PATH=f'{DATA_PATH}/hans_2024'
EXPERIMENT_CONFIGS_PATH=f'{PROJECT_ROOT_PATH}/experiments'
#LOG_PATH=f'{PROJECT_ROOT_PATH}/logs'
PERPLEXITY_OUTPUT_FILE_NAME='old_hans_perplexity_ratio_score_data.json'

# Logging stuff
LOG_LEVEL='DEBUG'
LOG_PREFIX='%(levelname)s - %(name)s - %(message)s'
CLEAR_LOGS=True

######################################################################
# Default model parameters ###########################################
######################################################################

# Loading details
CACHE_DIR='/mnt/fast_scratch/huggingface_transformers_cache'
HF_MODEL_STRING='meta-llama/Meta-Llama-3-8B'
MODEL_NAME='LLaMA3'
DEVICE_MAP='cuda:1'
CPU_CORES=8
AVAILABLE_GPUS=['cuda:0', 'cuda:1', 'cuda:2']

# Quantization configuration defaults
QUANTIZATION='4-bit'
BNB_4BIT_COMPUTE_DTYPE=torch.float16

# BNB warns that this doesn't exist... setting via bnb_4bit_compute_dtype?
#BNB_8BIT_COMPUTE_DTYPE=torch.float16

# Generation configuration defaults
MAX_NEW_TOKENS=32

# Decoding strategy
DECODING_STRATEGY=None

# Default test prompt for generation
PROMPT='It was a dark and stormy night '

######################################################################
# Perplexity ratio score stuff #######################################
######################################################################

# Parameters for the v2 scoring algorithm
WORKERS = 1
BATCH_SIZE = 10
WRITER_DEVICE = 'cuda:2'
READER_DEVICE = 'cuda:1'
CPUS_PER_WORKER = 4

SHORT_FRAGMENT_LIMIT = 10
LONG_FRAGMENT_LIMIT = 300

# Variable names to collect data for
DEPENDENT_VARS = [
    'Source record num',
    'Fragment length (words)',
    'Fragment length (tokens)',
    'Dataset',
    'Source',
    'Generator',
    'String',
    'Perplexity',
    'Cross-perplexity',
    'Perplexity ratio score',
    'Reader time (seconds)',
    'Writer time (seconds)',
    'Reader peak memory (GB)',
    'Writer peak memory (GB)',
]

# Models to use for perplexity scoring of Hans 2024 text samples
READER_MODEL = 'meta-llama/Llama-2-7b-hf'
WRITER_MODEL = 'meta-llama/Llama-2-7b-chat-hf'

# Paths dictionary to JSON lines data files from the
# original binoculars publication
HANS_DATA_FILES={
    'pubmed-falcon7': f'{HANS_DATA_PATH}/pubmed/pubmed-falcon7.jsonl',
    'pubmed-llama2-13': f'{HANS_DATA_PATH}/pubmed/pubmed-llama2_13.jsonl',
    'cnn-falcon7': f'{HANS_DATA_PATH}/cnn/cnn-falcon7.jsonl',
    'cnn-llama2-13': f'{HANS_DATA_PATH}/cnn/cnn-llama2_13.jsonl',
    'cc_news-falcon7': f'{HANS_DATA_PATH}/cc_news/cc_news-falcon7.jsonl',
    'cc_news-llama2-13': f'{HANS_DATA_PATH}/cc_news/cc_news-llama2_13.jsonl',
}

# Written by falcon-7b-instruct via bartleby
MACHINE_SUSPECT_STRING='''As the sun rose one day, a bird perched at the top of a
mountain and looked down onto a deep and murky swamp below. He felt alone and out 
of place, like he didn't belong in either the sky or the swamp. He had no one to 
talk to, no one to share his secret feelings with. As he looked around, he noticed 
a tiny turtle in the distance, and he decided to go down and make friends. As the 
bird and turtle spent time together, the bird learned that there are other creatures 
in the world like him, and that it doesn't always have to be him against the world.'''

# From my GitHub
HUMAN_SUSPECT_STRING='''Bartleby is a LLM based conversational collaborator written
in Python using HuggingFace transformers, discord.py, matrix-nio and google-api-core 
among others. The project goal is to create an open source, conversational writing 
assistant which interacts naturally via a chat interface and can generate documents 
in docx format. A 'universal' interface was achieved using Discord (i.e. the user can 
interact with the Bartleby via any Discord client application using any device: phone, 
laptop or tablet running: macOS, Windows, Linux, Android, IOS etc,). Bartleby can also 
interact via a Matrix server. Documents are created and accessed via Google Drive using 
Google Cloud Platform APIs.'''

ENCODING_TEST_TEXT='''Sir Isaac Newton FRS (25 December 1642 – 20 March 1726/27) was an
English polymath active as a mathematician, physicist, astronomer, alchemist, theologian, 
and author who was described in his time as a natural philosopher. He was a key figure 
in the Scientific Revolution and the Enlightenment that followed. His pioneering book 
Philosophiæ Naturalis Principia Mathematica (Mathematical Principles of Natural 
Philosophy), first published in 1687, consolidated many previous results and established 
classical mechanics Newton also made seminal contributions to optics, and shares credit 
with German mathematician Gottfried Wilhelm Leibniz for developing infinitesimal calculus, 
though he developed calculus years before Leibniz. In the Principia, Newton formulated the 
laws of motion and universal gravitation that formed the dominant scientific viewpoint for 
centuries until it was superseded by the theory of relativity. Newton used his mathematical 
description of gravity to derive Kepler's laws of planetary motion, account for tides, the 
trajectories of comets, the precession of the equinoxes and other phenomena, eradicating 
doubt about the Solar System's heliocentricity He demonstrated that the motion of objects 
on Earth and celestial bodies could be accounted for by the same principles. Newton's 
inference that the Earth is an oblate spheroid was later confirmed by the geodetic 
measurements of Maupertuis, La Condamine, and others, convincing most European scientists 
of the superiority of Newtonian mechanics over earlier systems. Newton built the first 
practical reflecting telescope and developed a sophisticated theory of color based on the 
observation that a prism separates white light into the colors of the visible spectrum. 
His work on light was collected in his highly influential book Opticks, published in 1704. 
He also formulated an empirical law of cooling, made the first theoretical calculation of 
the speed of sound, and introduced the notion of a Newtonian fluid. In addition to his work 
on calculus, as a mathematician Newton contributed to the study of power series, generalized 
the binomial theorem to non-integer exponents, developed a method for approximating the roots 
of a function, and classified most of the cubic plane curves. Newton was a fellow of Trinity 
College and the second Lucasian Professor of Mathematics at the University of Cambridge. 
He was a devout but unorthodox Christian who privately rejected the doctrine of the Trinity. 
He refused to take holy orders in the Church of England, unlike most members of the Cambridge 
faculty of the day. Beyond his work on the mathematical sciences, Newton dedicated much of 
his time to the study of alchemy and biblical chronology, but most of his work in those areas 
remained unpublished until long after his death. Politically and personally tied to the Whig 
party, Newton served two brief terms as Member of Parliament for the University of Cambridge, 
in 1689–1690 and 1701–1702. He was knighted by Queen Anne in 1705 and spent the last three 
decades of his life in London, serving as Warden (1696–1699) and Master (1699–1727) of the 
Royal Mint, as well as president of the Royal Society (1703–1727). Isaac Newton was born 
(according to the Julian calendar in use in England at the time) on Christmas Day, 25 December 
1642 (NS 4 January 1643) at Woolsthorpe Manor in Woolsthorpe-by-Colsterworth, a hamlet in the 
county of Lincolnshire His father, also named Isaac Newton, had died three months before. Born 
prematurely, Newton was a small child; his mother Hannah Ayscough reportedly said that he could 
have fit inside a quart mug. When Newton was three, his mother remarried and went to live with 
her new husband, the Reverend Barnabas Smith, leaving her son in the care of his maternal 
grandmother, Margery Ayscough (née Blythe). Newton disliked his stepfather and maintained some 
enmity towards his mother for marrying him, as revealed by this entry in a list of sins 
committed up to the age of 19: "Threatening my father and mother Smith to burn them and the 
house over them. Newton's mother had three children (Mary, Benjamin, and Hannah) from her 
second marriage. From the age of about twelve until he was seventeen, Newton was educated at 
The King's School in Grantham, which taught Latin and Ancient Greek and probably imparted a 
significant foundation of mathematics He was removed from school by his mother and returned 
to Woolsthorpe-by-Colsterworth by October 1659. His mother, widowed for the second time, 
attempted to make him a farmer, an occupation he hated.[18] Henry Stokes, master at The 
King's School, persuaded his mother to send him back to school. Motivated partly by a desire 
for revenge against a schoolyard bully, he became the top-ranked student distinguishing 
himself mainly by building sundials and models of windmills. In June 1661, Newton was admitted 
to Trinity College at the University of Cambridge. His uncle Reverend William Ayscough, who 
had studied at Cambridge, recommended him to the university. At Cambridge, Newton started as 
a subsizar, paying his way by performing valet duties until he was awarded a scholarship in 
1664, which covered his university costs for four more years until the completion of his MA. 
At the time, Cambridge's teachings were based on those of Aristotle, whom Newton read along 
with then more modern philosophers, including Descartes and astronomers such as Galileo 
Galilei and Thomas Street. He set down in his notebook a series of "Quaestiones" about 
mechanical philosophy as he found it. In 1665, he discovered the generalised binomial 
theorem and began to develop a mathematical theory that later became calculus. Soon after 
Newton obtained his BA degree at Cambridge in August 1665, the university temporarily closed 
as a precaution against the Great Plague. Although he had been undistinguished as a Cambridge 
student,Newton's private studies at his home in Woolsthorpe over the next two years saw 
the development of his theories on calculus, optics, and the law of gravitation. In April 
1667, Newton returned to the University of Cambridge, and in October he was elected as a 
fellow of Trinity Fellows were required to take holy orders and be ordained as Anglican 
priests, although this was not enforced in the Restoration years, and an assertion of 
conformity to the Church of England was sufficient. He made the commitment that "I will 
either set Theology as the object of my studies and will take holy orders when the time 
prescribed by these statutes [7 years] arrives, or I will resign from the college. Up 
until this point he had not thought much about religion and had twice signed his agreement 
to the Thirty-nine Articles, the basis of Church of England doctrine. By 1675 the issue 
could not be avoided, and by then his unconventional views stood in the way. His academic 
work impressed the Lucasian professor Isaac Barrow, who was anxious to develop his own 
religious and administrative potential (he became master of Trinity College two years 
later); in 1669, Newton succeeded him, only one year after receiving his MA. The terms 
of the Lucasian professorship required that the holder not be active in the church – 
presumably to leave more time for science. Newton argued that this should exempt him from 
the ordination requirement, and King Charles II, whose permission was needed, accepted 
this argument; thus, a conflict between Newton's religious views and Anglican orthodoxy 
was averted. Newton's work has been said "to distinctly advance every branch of mathematics 
then studied”. His work on the subject, usually referred to as fluxions or calculus, seen 
in a manuscript of October 1666, is now published among Newton's mathematical papers. 
His work De analysi per aequationes numero terminorum infinitas, sent by Isaac Barrow to 
John Collins in June 1669, was identified by Barrow in a letter sent to Collins that 
August as the work "of an extraordinary genius and proficiency in these things”. Newton 
later became involved in a dispute with Leibniz over priority in the development of calculus. 
Most modern historians believe that Newton and Leibniz developed calculus independently, 
although with very different mathematical notations. However, it is established that Newton 
came to develop calculus much earlier than Leibniz. Leibniz's notation and "differential 
Method", nowadays recognized as much more convenient notations, were adopted by continental 
European mathematicians, and after 1820 or so, also by British mathematicians.'''
