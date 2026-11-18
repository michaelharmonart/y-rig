---
weight: 20
title: "y-rig: Our Approach"
description: ""
icon: "layers"
date: "2026-10-17T17:54:57-06:00"
lastmod: "2026-10-17T17:54:57-06:00"
toc: true
---

# Why rig the y-rig way?

While y-rig is by no means a perfect system, we do feel confident in the overall approach it takes and philosophy behind it.
This confidence comes from an understanding of a pattern and path of progression that many individuals and entire studios have already experienced.

Generally it goes something like this:

## Chapter 1: Rigging by Hand

Creating controls, deformers, and wiring them up is the bread and butter of rigging. \
Unfortunately most of the rigging software out there is highly specialized and granular; you can control everything,
but doing simple things like making a single control and hooking it to a joint takes a number of steps.
After you've performed the same sequence of tasks a few dozen times it's a natural progression to a shift in approach.

## Chapter 2: Helper Scripts

If taking a control takes so many steps to do, but most of the steps are the same every time... just script it!
Maybe you write a little function that takes a transform as input and creates an entire control, names it, and hooks it to a joint in one step.
Suddenly rigging is a lot less tedious, and you are free to focus on the things that make each problem unique.
But before you know it a similar problem creeps back in. Let's say you've set up an entire arm rig: fancy IK/FK switching, stretch and twist, etc.
That's at least a few dozen steps in any modern rigging package. What if you want to make another arm for the right side?
Or maybe an almost identical limb setup for a leg? Too bad, you're going to have to repeat a lot of work.
It's time for another shift in approach.

## Chapter 3: Scripted Components

Really what you realize you want is the ability to re-use little pieces/components throughout your rigs.
Since Maya, Blender and friends don't really have a concept of rigging with re-usable components,
the easiest solution is to expand your scripts to have the ability to build entire components.
Maybe you have a `limb` class that can build a basic limb, and an `arm` and `leg` component that inherit from it.
With this approach you only have to define/build your actual rig description once: you author a python script instead of a Maya scene,
and then you can re-use that component wherever you need by just running a line or two of code
and passing in any parameters you need to specifcy for your particular use of that component.

This works reasonably well, so problem solved right? Well, not quite...

Turns out that the same problem of having a little script to build a control,
but then having to do the same setup of hooking all those controls together everytime you want to actually
build a full limb is now not so different from the new problem:
You still have numerous (maybe _hundreds_) of steps that are required
to actually create all the components of a basic character and hook them all together.

Looks like it's time for another shift in approach.

## Chapter 4: The Build Script

If most characters require about the same sort of base rig (and they do),
and the process of hooking the components together and setting up a full rig from start to finish is so tedious,
let's just automate the whole thing!

The system in use before y-rig operated like this.
There was one large python script that defined the entire process of building a character from start to finish.
This removes a lot of chances for forgotten steps, human error, broken rigs, etc.
It also makes it much easier to deal with the fact that things have to change! Model gets updated? Just change the file the script imports the model from,
and it'll bind all the same joints and deformers in the exact order needed! Need to make a change to how all the limbs work such as adding bend controls?
Just update the limb builder and re-run the master build script and all four of the characters limbs have bend controls now!

Sounds like a pretty nice system. It is! But before long we run into similarly shaped problems as before.
What if we want to have 2 human characters instead of one? What if we want to also have a quadruped?
The easy solution is to just start adding some logic breaks or if statements to the rig builder.
If it's a quadruped the spine needs oriented on x instead of y, if it's the female protagonist we build the hair controls, etc.
This works well enough at first but quickly becomes a nightmare to deal with.
We used such a system for four capstone projects and by that fourth project the `build.py` file had reached THOUSANDS of lines,
and making any change to one character might accidentally break things for a dozen others.

Sounds like it's time to shift our approach.

## Chapter 5: The Modular Build System

Instead of one big file where character specifics are defined in code,
lets instead have a central system that handles the basic structure of a rig,
and then have a set of "guides" or a "metarig", which is what actually defines the rig. Our build system then reads that,
runs the necessary code to build the components, hook them together, and then maybe run some custom rig specific code before or after the build.

This is approach of most rigging solutions out there. mGear, Blender's rigify, Advanced Skeleton, Zoo Tool's Hive, and others.
This approach means we can have some sort of shared base (the components and build system) which are reusable between characters,
while not having a mess of character specific logic polluting the core of the rigging system.

This works pretty well! If it didn't most people would keep going instead of stopping here.
But soon enough we run into the same sort of problem as we did before.
Once you've set up 5, 10, 25 characters each with their own character file and custom scripts, you realize you're duplicating work.
What if you wanted to change all the human characters to have dual quaternion skin weights but you want to keep the props linear skinned?
You could do this in the main build system, but now you're falling into the same trap as all those if statements in the `build.py`
approach we had in chapter 4.

Tricky business...
I reckon this smells like it's time to shift our approach!

## Chapter 6: Real Composition

From the very get-go we've been trying to solve a very simple problem:

> "I did something once, I just want to re-use it with a few tweaks somewhere else."

Turns out this is a problem as old as computer animation, and people much smarter than me have already proposed some pretty great solutions.
We've borrowed our approach from Pixar and how they handle scene description. You've probably heard of USD. If you haven't now is a great time to change that.
USD is at its core a pretty simple solution to the above problem of re-using work with tweaks.
When you get a chance you should read the [Introduction to USD](https://openusd.org/release/intro.html)

But USD is good for the same reason pretty much anything gets good: iterative refinement.

> USD is roughly the fourth generation of “composed scene description” developed at Pixar.
> After muscling through Toy Story, in which each shot was described by a single, linear program file,
> the Pixar R&D team began adding and evolving concepts for referencing, layering, editing, and variation
> in the context of its proprietary animation system, Marionette (known internally as Menv),
> beginning with A Bug’s Life, and continuing over the course of the next ten feature films.

From the USD [Introduction to USD](https://openusd.org/release/intro.html#heritage-of-usd-at-pixar)

Their work paid off. USD caught hold of the industry and spread like wildfire because it solved such a fundamental problem so simply.
You have something you want to re-use somewhere else? Just reference that USD file.
Need to make some edits on top that are specific to whatever you're working on? Just add another layer and make some overrides on just what you need.

This is an an _incredibly_ good solution for the type of problem we've been trying to solve with all these rigging approaches.
With the USD composition system we can make our own little components, use them wherever need them, combine them into bigger components,
and override something as minute as a single control all the way up at the layer of an entire character. This is an impossible task with any of the other approaches.
If we have a python class for a limb, the farthest we can hope to override is making a subclass and swapping a variable or an entire function out.

It also opens up entirely new ways of working. We can have all human characters inherit from a base biped,
and then override only what is needed for each individual character.
We can have a character inherit from another unrelated character.
Maybe you like the staff rig your colleague rigged for a character on a different project,
just reference it into your character rig and make some tweaks. No data is duplicated and you can make sweeping changes when needed.
Found a bug in how the ankles behave in a specific IK mode, and it affects all characters?
No worries, just fix it in the leg component's file and the fix is applied everywhere.

Unfortunately because USD is meant to be a _Universal_ Scene Description,
it is missing many of the features required for actually authoring rigs the way that was just outlined.
Pixar plans to add more of their rigging functionality from Presto into USD, but it will be many years until we can use it for all our rigging needs.

## Chapter 7: y-rig

Luckily we don't have to wait. The principles in effect in USD can be applied to rigging in Maya.
Obviously Maya doesn't have all of the composition systems that USD does,
references and overrides to references exist but are very primitive and pretty much entirely unusable for rigging,
layers exist only for animation data, and things like inheritance, variants, etc don't exist at all.
But, while Maya may not have these features it _is_ a very capable and robust execution engine and rigging package.
All of the building blocks are there, just none of the composition arcs to combine them effectively.

Turns out this problem has also been solved before. Computers have a set of very simple small building blocks (instructions, things they can do)
which strung together can do incredible things. But actually writing every single instruction by hand would be extremely tedious.
So, some people who are much smarter than me realized you could actually have a different language of instructions (a programming language)
and then convert/compile that language into something the computer can actually process.
You could design your system to use whatever principles you wanted,
and still have the final result be able to run on those dumb little computers with only their simple little instructions.
We can design our system to have all the benefits of USD and still make rigs that run on dumb little Maya with only its simple little nodes.

We do this by authoring rigs using a higher-level description (one with the many of the niceties of USD), and build the result into a regular Maya scene.
The tool we've picked for this is NXT. Please read the basics of NXT [here](https://nxt-dev.github.io/concepts/). You'll notice it's very similar to USD!

At this point hopefully the idea has clicked. We can finally define a core building block once, use it anywhere, and tweak only what we need.
We just need the tooling to do it.

## Chapter 8: The Future (and you)

Now that you know this story, I hope that you will not have to suffer through trying to solve the same
core problem so many have worked to solve before, only to realize you've re-invented a bad wheel.
As you decide what to change about y-rig, please be smart.
There is much that needs to be changed, but my hope is that you understand what is worth keeping and what is worth reworking or replacing.
Hopefully new and improved tools and opportunities will make their way into the world and make some of our custom solutions obsolete.
I didn't set out to write an entirely custom rigging system, but unfortunately there was nothing out there built on the correct approach and foundation.
If you choose to change or replace something, please ensure that you do not regress to approaches outlined in the previous chapters.
If you do, you will also suffer the pains to go with it.
For the time being, USD is in my mind almost a perfect solution. Steal from it wherever possible.

I hope the foundation here will allow you to focus on the art of rigging,
and on solving new and interesting problems, rather than doing tedious work you know you've done before.

Good luck out there!
