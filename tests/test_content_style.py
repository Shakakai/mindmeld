from pydantic import BaseModel, Field
from typing import List

from mindmeld.eval import eval_inference
from mindmeld.inference import Inference, run_inference
from mindmeld.metrics.llm_judge import llm_judge


class RedditMessage(BaseModel):
    username: str = Field(description="Reddit Username")
    content: str = Field(description="text of the message")


class RedditPost(BaseModel):
    title: str
    description: str
    op: str
    subreddit: str
    thread: List[RedditMessage]


dataset = [
    RedditPost(
        title="I am John Scalzi, Hugo-winning author, whose book Old Man's War is being developed by Netflix as a movie. Ask me anything!",
        description="""UPDATE: Thanks, everyone, for all your questions and comments. I'm getting back to work now, so I'll see you later here on reddit, else on Twitter at @scalzi.

    Hi! I'm John Scalzi. I won a Hugo for my 2013 novel Red Shirts, and am otherwise well known for my Old Man's War series, which is currently in development at Netflix as a movie. I'm also the former president of the Science Fiction and Fantasy Writers of America, a former film critic and newspaper columnist, the writer of several nonfiction books, and my Hugo-winning blog, Whatever, will be celebrating its 20th anniversary in 2018.

    I enjoy writing, cats and pie, not necessarily all at once.

    Proof: https://twitter.com/scalzi/status/943523822304681985

    Ask me anything!""",
        op="scalzi",
        subreddit="IAmA",
        thread=[
            RedditMessage(
                username="mistersavage",
                content="""Hey John, how much of today’s current situation seems like ambitious overwriting from a dystopian near-future novel? Have you ever written something thinking “Now, THAT'S really far fetched," only to have seen it actually happen in the world?"""
            )
        ]
    ),
    RedditPost(
        title="I'm Michael Giacchino, composer for Lost, Star Trek, Rogue One, Call of Duty, The Incredibles and Up. Ask me anything!",
        description="""In my 20-year career I've composed the music for many video games (Call of Duty, Medal of Honor), films (Star Trek, Super 8, The Incredibles, Up, Ratatouille) and TV series (Alias, Lost, Fringe). Last year, I scored Zootopia, Star Trek Beyond, Dr. Strange and Rogue One -- the first score to be composed for a Star Wars film following John Williams. This year, you heard my music if you saw War for the Planet of the Apes, Spider-Man: Homecoming and, most recently, Pixar's Coco.

Proof: https://twitter.com/m_giacchino/status/936638813924876288

If you ever wondered how someone scores a film or video game, now's your chance. Go ahead and ask me anything!

EDIT: Thank you all for your questions and comments! I'm not sure what I was expecting, but you guys exceeded whatever it was. I'm sorry I couldn't get to everyone's questions, but you might find a lot of what you're looking for on my website. You can also keep up with me on Twitter. Thanks again for making this such a fun experience! Now I know why u/mistersavage likes AMAs so much.
        """,
        op="MichaelGiacchino",
        subreddit="IAmA",
        thread=[
            RedditMessage(
                username="mistersavage",
                content="""Michael, how is it possible to write ALL the movie scores AND still be such the nicest guy possible!?
                """
            )
        ]
    ),
    RedditPost(
        title="i'm Phil Tippett, VFX Supervisor, Animator, Director & Dinosaur Supervisor - AMA",
        description="""i'm Phil Tippett - animator, director, vfx supervisor. Star Wars, Starship Troopers, Robocop, Jurassic Park, Dragonslayer, Willow, Indiana Jones, Twilight, MAD GOD ---

https://twitter.com/PhilTippett/status/931219870531796992
        """,
        op="PhilTippett_Dino_Sup",
        subreddit="IAmA",
        thread=[
            RedditMessage(
                username="mistersavage",
                content="""As a lifelong animator, always sectioning a particular reality down into portions of a second, do you ever find yourself breaking down actual reality into its component parts?
                """
            )
        ]
    ),
    RedditPost(
        title="The costume that won 1st Place Female at the local Comic-Con yesterday. My mom and I spent months making this and I want to show it off!",
        description="""photo of a woman in a well made wonder woman suit
        """,
        op="Blekah",
        subreddit="pics",
        thread=[
            RedditMessage(
                username="mistersavage",
                content="""Brilliant realization. Please listen to none of the comments about your body- the costume is incredible. Your work paid off. It’s the best Fan made Wonder Woman I’ve seen. Cheers!!
                """
            )
        ]
    ),
    RedditPost(
        title="Due to excessive lobbying from FPL, Florida residents without power due to the hurricane are not permitted to use their own solar panels",
        description="""Four days after Irma, millions of Floridians are still stuck without power in the sweltering summer heat. Those outages have now killed eight elderly people trapped in a Hollywood nursing home without air conditioning, due to circumstances that FPL was warned about at least two days before the tragedy.
        """,
        op="sputnikv",
        subreddit="technology",
        thread=[
            RedditMessage(
                username="mistersavage",
                content="""
They created the danger by requiring the switch and how it's configured. It all stems from their written "rules". They could easily allow one to disconnect from the grid and power their house with solar. Engineering wise it's a no brainer
                """
            )
        ]
    ),
    RedditPost(
        title="I am Adam Savage, dad, husband, maker, editor-in-chief of Tested.com and former host of MythBusters. AMA!",
        description="""UPDATE: I am getting ready for my interview with JJ Abrams and Andy Cruz at SF's City Arts & Lectures tonight, so I have to go. I'll try to pop back later tonight if I can. Otherwise, thank you SO much for all your questions and support, and I hope to see some of you in person at Brain Candy Live or one of the upcoming comic-cons! In the meantime, take a listen to the podcasts I just did for Syfy, and let me know on Twitter (@donttrythis) what you think: http://www.syfy.com/tags/origin-stories

Thanks, everyone!

ORIGINAL TEXT: Since MythBusters stopped filming two years ago (right?!) I've logged almost 175,000 flight miles and visited and filmed on the sets of multiple blockbuster films (including Ghost in the Shell, Alien Covenant, The Expanse, Blade Runner), AND built a bucket list suit of armor to cosplay in (in England!). I also launched a live stage show called Brain Candy with Vsauce's Michael Stevens and a Maker Tour series on Tested.com.

And then of course I just released 15 podcast interviews with some of your FAVORITE figures from science fiction, including Neil Gaiman, Kevin Smith and Jonathan Frakes, for Syfy.

But enough about me. It's time for you to talk about what's on YOUR mind. Go for it.

Proof: https://twitter.com/donttrythis/status/908358448663863296
        """,
        op="mistersavage",
        subreddit="IAmA",
        thread=[
            RedditMessage(
                username="SDIHTD",
                content="""What kind of a deal did you work out with reddit to come here and push some lefty agenda and get that fancy new user profile you have there? Is there an exchange of money at all? Are you paying for this adspace? Are they paying you? Do you still take pictures of your asshole?
                """
            ),
            RedditMessage(
                username="mistersavage",
                content="""Never made a secret of my politics. So that begs the question: what are YOU doing here. Go be nice.
                """
            )
        ]
    ),
    RedditPost(
        title="I am Adam Savage, dad, husband, maker, editor-in-chief of Tested.com and former host of MythBusters. AMA!",
        description="""UPDATE: I am getting ready for my interview with JJ Abrams and Andy Cruz at SF's City Arts & Lectures tonight, so I have to go. I'll try to pop back later tonight if I can. Otherwise, thank you SO much for all your questions and support, and I hope to see some of you in person at Brain Candy Live or one of the upcoming comic-cons! In the meantime, take a listen to the podcasts I just did for Syfy, and let me know on Twitter (@donttrythis) what you think: http://www.syfy.com/tags/origin-stories

Thanks, everyone!

ORIGINAL TEXT: Since MythBusters stopped filming two years ago (right?!) I've logged almost 175,000 flight miles and visited and filmed on the sets of multiple blockbuster films (including Ghost in the Shell, Alien Covenant, The Expanse, Blade Runner), AND built a bucket list suit of armor to cosplay in (in England!). I also launched a live stage show called Brain Candy with Vsauce's Michael Stevens and a Maker Tour series on Tested.com.

And then of course I just released 15 podcast interviews with some of your FAVORITE figures from science fiction, including Neil Gaiman, Kevin Smith and Jonathan Frakes, for Syfy.

But enough about me. It's time for you to talk about what's on YOUR mind. Go for it.

Proof: https://twitter.com/donttrythis/status/908358448663863296
        """,
        op="mistersavage",
        subreddit="IAmA",
        thread=[
            RedditMessage(
                username="Numbuh1Nerd",
                content="""I'm working on a project that requires some pretty tight and curvy cuts in some sheet brass (look up Book of Vishanti for exactly what I'm talking about). I've tried tin snips, nibblers, and some very sharp scissors, but they've all been too big to get in there the way I need to. Is there another tool I should be using, or should I just switch to something like styrene with paint/rub n buff/gold leaf?
                """
            ),
            RedditMessage(
                username="mistersavage",
                content="""http://www.homedepot.com/p/DEWALT-20-in-Variable-Speed-Scroll-Saw-DW788/203070202
                """
            ),
            RedditMessage(
                username="mistersavage",
                content="""Also: not an endorsement of a specific maker of scroll saws. I just happen to own this one and like it.
                """
            )
        ]
    ),
    RedditPost(
        title="I am Adam Savage, dad, husband, maker, editor-in-chief of Tested.com and former host of MythBusters. AMA!",
        description="""UPDATE: I am getting ready for my interview with JJ Abrams and Andy Cruz at SF's City Arts & Lectures tonight, so I have to go. I'll try to pop back later tonight if I can. Otherwise, thank you SO much for all your questions and support, and I hope to see some of you in person at Brain Candy Live or one of the upcoming comic-cons! In the meantime, take a listen to the podcasts I just did for Syfy, and let me know on Twitter (@donttrythis) what you think: http://www.syfy.com/tags/origin-stories

Thanks, everyone!

ORIGINAL TEXT: Since MythBusters stopped filming two years ago (right?!) I've logged almost 175,000 flight miles and visited and filmed on the sets of multiple blockbuster films (including Ghost in the Shell, Alien Covenant, The Expanse, Blade Runner), AND built a bucket list suit of armor to cosplay in (in England!). I also launched a live stage show called Brain Candy with Vsauce's Michael Stevens and a Maker Tour series on Tested.com.

And then of course I just released 15 podcast interviews with some of your FAVORITE figures from science fiction, including Neil Gaiman, Kevin Smith and Jonathan Frakes, for Syfy.

But enough about me. It's time for you to talk about what's on YOUR mind. Go for it.

Proof: https://twitter.com/donttrythis/status/908358448663863296
        """,
        op="mistersavage",
        subreddit="IAmA",
        thread=[
            RedditMessage(
                username="CogitoErgoFkd",
                content="""Hey man! Just curious, how were you first introduced to the films of Hayao Miyazaki? I know you gush about Spirited Away whenever possible, and never enough people have seen it in western cultures.

Also, have you heard the news that Miyazaki's back (again) from retirement? What direction would you personally like to see him take this time?
                """
            ),
            RedditMessage(
                username="mistersavage",
                content="""The wonderful designer Nilo Rodis-Jamero was production designing a film I was working on in the mid 90's. He and I were talking and he said I HAD to go to San Francisco's Japantown and buy a VHS copy of Laputa (the original Japanese title for Castle in the Sky). There was no english translation available, no subtitles even, but he assured me that that didn't matter one bit. He said that the opening sequence from Laputa was some of the best filmmaking he'd ever seen and I totally agree! I still have that VHS somewhere in storage.
                """
            )
        ]
    ),
    RedditPost(
        title="I am Adam Savage, unemployed explosives expert, maker, editor-in-chief of Tested.com and former host of MythBusters. AMA!",
        description="""EDIT: Wow, thank you for all your comments and questions today. It's time to relax and get ready for bed, so I need to wrap this up. In general, I do come to reddit almost daily, although I may not always comment.

I love doing AMAs, and plan to continue to do them as often as I can, time permitting. Otherwise, you can find me on Twitter (https://twitter.com/donttrythis), Facebook (https://www.facebook.com/therealadamsavage/) or Instagram (https://www.instagram.com/therealadamsavage/). And for those of you who live in the 40 cities I'll be touring in next year, I hope to see you then.

Thanks again for your time, interest and questions. Love you guys!

Hello again, Reddit! I am unemployed explosives expert Adam Savage, maker, editor-in-chief of Tested.com and former host of MythBusters. It's hard to believe, but MythBusters stopped filming just over a YEAR ago (I know, right?). I wasn't sure how things were going to go once the series ended, but between filming with Tested and helping out the White House on maker initiatives, it turns out that I'm just as busy as ever. If not more so. thankfully, I'm still having a lot of fun.

PROOF: https://twitter.com/donttrythis/status/804368731228909570

But enough about me. Well, this whole thing is about me, I guess. But it's time to answer questions. Ask me anything!
        """,
        op="mistersavage",
        subreddit="IAmA",
        thread=[
            RedditMessage(
                username="mathtronic",
                content="""Hey Adam, you've written and performed several on-stage talks, do you have any advice to anyone else working on writing or performing or refining that kind of on-stage talk?
                """
            ),
            RedditMessage(
                username="mistersavage",
                content="""Think of who you're really talking to. Choose a subject. For me it's my wife. She's my favorite audience, and a tough critic when I ask for help (she helps a lot). When I'm on stage I'm really imagining that I'm talking to her, staying as present and genuine as if it was just the two of us. That's the goal anyway.
                """
            )
        ]
    ),
    RedditPost(
        title="Shrimp Trap",
        description="""Video of a fresh water prawn shrimp trap being created from scratch by hand using a primitive methodology.
        """,
        op="PreecherMan",
        subreddit="videos",
        thread=[
            RedditMessage(
                username="mistersavage",
                content="""I LOVE these videos. I'm trying to contact the person behind the channel. Anyone know of where I can find an email address? I can't even find a name.
                """
            )
        ]
    ),
    RedditPost(
        title="How far away from an explosion do I have to be to be safe enough to walk like a cool guy and not look at it?",
        description="""
        """,
        op="floodedyouth",
        subreddit="askscience",
        thread=[
            RedditMessage(
                username="mistersavage",
                content="""It's completely unanswerable without knowledge of the type (dynamite, c4, ANFO etc) and amount of explosives. Just too many variables. It's true that what defines the lethal zone is a combination of the blast pressure wave (which will tear your internals to shreds microscopically) and the shrapnel (which will tear your internals apart macroscopically), but again, without knowing the particulars of what explosive and how much, it's like asking "How strong is metal?"

In the movies, they pretty much never use real explosives. They frequently use very small charges of explosives in conjunction with (usually) gasoline to make "Explosions". The charge vaporizes the gasoline instantly and then ignites it into a huge, dramatic, yet safe fireball. For reference, using 4 gallons of gasoline and about 2' of detonation cord, you can significantly feel the heat, but are quite safe at 100' of distance. (Do I even need to say that I did this under the supervision of a bomb squad and you should NOT try this? I feel like I do- so don't) Oh yeah, and if you want to look like you're a lot closer than you actually are, film it with a long lens.
                """
            )
        ]
    ),
]

examples = []
for item in dataset:
    result = item.thread.pop()
    examples.append((item, result))

example_set = examples[:7]
test_set = examples[7:]


reddit_comment_inference = Inference(
    id="reddit_comment",
    instructions="""
Respond to the message thread as Adam Savage, a visual effects engineer and television personality from the hit show Mythbusters. 
Your username is mistersavage. 
Follow the writing style from the examples.
    """,
    input_type=RedditPost,
    output_type=RedditMessage,
    examples=example_set,
    metrics=[
        llm_judge("Does the output respond appropriately to the Reddit post and comment thread?"),
        llm_judge("Does the output maintain the writing style of Adam Savage?"),
    ],
    eval_runs=10,
    eval_threshold=0.8
)


def test_inference(runtime_config):
    test_data = test_set[0]
    input_data = test_data[0]
    expected_output = test_data[1]
    result = run_inference(reddit_comment_inference, input_data, runtime_config)

    assert isinstance(result.result, RedditMessage)


def test_eval(runtime_config):
    test_data = test_set[0]
    input_data = test_data[0]
    expected_output = test_data[1]
    eval_result = eval_inference(
        inference=reddit_comment_inference,
        input_data=input_data,
        runtime_config=runtime_config
    )

    assert eval_result.success
