# autoenc-neurons (work in progress)

An experiment with convolutional autoencoders as applied to text, and exploring what the
'neurons' in the convolutional layers respond to.

## Running

Note that if you do not have GPUs, you will need to edit the script to install `tensorflow` 
instead of `tensorflow-gpu`, and change `gpu_count` to 0.

```bash
python3 autoenc-neurons.py
```

This will start model training and then produce output like this:

```
very fine letter of ample explanation, regret, and entreaty, to his
   .--........  . .-..-......-.. - ...-.. . ..=...... ..  ..-. .- ..------------
.. --. .. .  -.... . . . .   . - ..  --- .-  -- ...  --. .--.-... ..............
  .-- -.........= .-..--.....-..... .-..-..-...........-...-.. .. -.............
.-  . -.. ....-..-  ..  . .   ... --....--............... ... -..=............. 
   .-..... . .. .-..-.  -...  -.. -..-=    . -=...... . ...-=.-. -..------------
 .-...= .- -.  .....-.- . ... = =  -... .  - ..=.. =-   =...= .... .............
-.-  = ...-...-..-. ......-... ..... ..=.--........-..-.... .-..= . ............
 .-..   -. .. .=.--. .  --..        .-.......-.  ....  ..  -..-- .-============-
 ....-......   ...--... ...... ..  .........  ...-...  ..  ..  -. ..............
 ...-. -.. .....- .. ....- .. .....-.--. .....- .-- .-.  ..--..-  --............
very fine letter of amlle eelealation, geeret, and entreatyy to his   
```

This begins with a line from the text, then shows activations that the text caused
in several of the filters or 'neurons' in the first convolutional layer. 
It's crude. Bigger/darker characters means more active. The last line is the 
autoencoder's reconstruction of the input. For example, the sixth filter seems
to respond to vowels, especially 'a'.

It's later followed by output like:

```
very fine letter of ample explanation, regret, and entreaty, to his
....--..==..--..==....--..--..  ------==......--  --......--==..------........--
..==..------**......--==..--==--==..--....====..--..==..**--==..............    
----....--....----....==....----==--------==--  ------==....  ........          
  ..------------..--..------....--..----....--..--......  --==------==========--
..----........--..----......--..--......  ......    ..----------..------------..
----..----....----==..  ----..    ..==--....==--------....==--==--====********==
....**......--------....--    ----..--==------==--==......--==--....------------
....--..==..--    --..--....--..==------  --..--..==----....--..----..          
--  ----..--==--......----..----------==....----------==..==--==--..............
------......--..==  ..    ..--....==--------==--....--..----==..==............  
very fine letter of amlle eelealation, geeret, and entreatyy to his   
```

These are activations from the second convolutional layer. For example, the second
filter seems to respond to a consonant plus 't'.

## References

The source text is The Complete Project Gutenberg Works of Jane Austen:                       
http://www.gutenberg.org/files/31100/31100.txt

For more on convolutional autoencoders in Keras, see:
https://blog.keras.io/building-autoencoders-in-keras.html