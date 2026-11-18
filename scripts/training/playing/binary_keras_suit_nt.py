import itertools
import os
import os.path
import numpy as np

from binary_util import play_data_iterator, get_play_labels, DealMeta, seats, seat_index, suit_index_lookup, binary_hand, get_card_index, hot_encode_card, wins_trick_index

def lead_holder(hands, play_str):
    # The seat (index in the deal string) that holds the card led to the first trick
    suit, rank = suit_index_lookup[play_str[0]], play_str[1]
    return next((i for i, hand in enumerate(hands) if rank in hand[suit]), None)

def declarer_index(hands, d_meta, play_str):
    # The declarer is the right-hand opponent of the player who holds the opening lead. The
    # declarer letter of the record is not used: in the WBC file (data/WBC/play.txt) it agrees
    # with the cards played in only one record out of four, while the play itself fits the
    # hands in 99.9% of the records (check_play_data.py). With the letter, three quarters of
    # the training rows showed the network hands that did not hold the cards being played.
    holder = lead_holder(hands, play_str)
    if holder is None:
        return seat_index[d_meta.declarer]
    return (holder - 1) % 4

def play_fits_hands(deal_str, outcome_str, play_str):
    # True when every card of the play is in the hand whose turn it is, from the real leader on
    hands = [set(suit + card for suit, cards in zip('SHDC', hand_str.split('.')) for card in cards)
             for hand_str in deal_str[2:].split()]
    cards = [play_str[i:i + 2] for i in range(0, len(play_str), 2)]
    if len(cards) != 52 or len(hands) != 4:
        return False
    on_lead = next((i for i, hand in enumerate(hands) if cards[0] in hand), None)
    if on_lead is None:
        return False
    trump = DealMeta.from_str(outcome_str).strain
    for t in range(0, 52, 4):
        trick = cards[t:t + 4]
        for i, card in enumerate(trick):
            hand = hands[(on_lead + i) % 4]
            if card not in hand:
                return False
            hand.remove(card)
        on_lead = wins_trick_index(trick, trump, on_lead)
    return True

def binary_data_righty(deal_str, outcome_str, play_str):
    '''
    ## input
      0:32  player hand
     32:64  seen hand (dummy, or declarer if we create data for dummy)
     64:96  last trick card 0
     96:128 last trick card 1
    128:160 last trick card 2
    160:192 last trick card 3
    192:224 this trick lho card
    224:256 this trick pard card
    256:288 this trick rho card
    288:292 last trick lead player index one-hot
        292 level
    293:298 strain one hot N, S, H, D, C

    ## output
    0:32 card to play in current trick
    '''

    x = np.zeros((1, 11, 298), np.float16)
    y = np.zeros((1, 11, 32), np.float16)
    
    hands = list(map(lambda hand_str: list(map(list, hand_str.split('.'))), deal_str[2:].split()))
    d_meta = DealMeta.from_str(outcome_str)
    declarer_i = declarer_index(hands, d_meta, play_str)
    leader_i = (declarer_i + 1) % 4
    dummy_i = (declarer_i + 2) % 4
    me_i = (dummy_i + 1) % 4

    dummy_bin = binary_hand(hands[dummy_i])
    me_bin = binary_hand(hands[me_i])
    
    _, on_leads, last_tricks, cards_ins, card_outs = get_play_labels(play_str, d_meta.strain, 2)
    
    dummy_played_cards = set(['>>'])
    
    for i, (on_lead, last_trick, cards_in, card_out) in enumerate(zip(on_leads, last_tricks, cards_ins, card_outs)):
        if i > 10:
            break
        label_card_ix = get_card_index(card_out)
        y[0, i, label_card_ix] = 1
        x[0, i, 292] = d_meta.level
        if d_meta.strain == 'N':
            x[0, i, 293] = 1
        else:
            x[0, i, 294 + suit_index_lookup[d_meta.strain]] = 1
        
        x[0, i, 288 + on_lead] = 1
        
        last_trick_dummy_card = last_trick[1]
        if last_trick_dummy_card not in dummy_played_cards:
            dummy_bin[get_card_index(last_trick_dummy_card)] -= 1
            dummy_played_cards.add(last_trick_dummy_card)
        
        if cards_in[2] not in dummy_played_cards:
            dummy_bin[get_card_index(cards_in[2])] -= 1
            dummy_played_cards.add(cards_in[2])
        
        x[0, i, 32:64] = dummy_bin
        x[0, i, 0:32] = me_bin
        
        x[0, i, 64:96] = hot_encode_card(last_trick[0])
        x[0, i, 96:128] = hot_encode_card(last_trick[1])
        x[0, i, 128:160] = hot_encode_card(last_trick[2])
        x[0, i, 160:192] = hot_encode_card(last_trick[3])
        
        x[0, i, 192:224] = hot_encode_card(cards_in[0])
        x[0, i, 224:256] = hot_encode_card(cards_in[1])
        x[0, i, 256:288] = hot_encode_card(cards_in[2])
        
        me_bin[label_card_ix] -= 1
        
    return x, y

def binary_data_decl(deal_str, outcome_str, play_str):
    x = np.zeros((1, 11, 298), np.float16)
    y = np.zeros((1, 11, 32), np.float16)
    
    hands = list(map(lambda hand_str: list(map(list, hand_str.split('.'))), deal_str[2:].split()))
    
    d_meta = DealMeta.from_str(outcome_str)

    declarer_i = declarer_index(hands, d_meta, play_str)
    dummy_i = (declarer_i + 2) % 4
    me_i = declarer_i

    dummy_bin = binary_hand(hands[dummy_i])
    me_bin = binary_hand(hands[me_i])
    
    _, on_leads, last_tricks, cards_ins, card_outs = get_play_labels(play_str, d_meta.strain, 3)
    
    dummy_played_cards = set(['>>'])
    
    for i, (on_lead, last_trick, cards_in, card_out) in enumerate(zip(on_leads, last_tricks, cards_ins, card_outs)):
        if i > 10:
            break
        label_card_ix = get_card_index(card_out)
        y[0, i, label_card_ix] = 1
        x[0, i, 292] = d_meta.level
        if d_meta.strain == 'N':
            x[0, i, 293] = 1
        else:
            x[0, i, 294 + suit_index_lookup[d_meta.strain]] = 1
        
        x[0, i, 288 + on_lead] = 1
        
        last_trick_dummy_card = last_trick[1]
        if last_trick_dummy_card not in dummy_played_cards:
            dummy_bin[get_card_index(last_trick_dummy_card)] -= 1
            dummy_played_cards.add(last_trick_dummy_card)
        
        if cards_in[1] not in dummy_played_cards:
            dummy_bin[get_card_index(cards_in[1])] -= 1
            dummy_played_cards.add(cards_in[1])
        
        x[0, i, 32:64] = dummy_bin
        x[0, i, 0:32] = me_bin
        
        x[0, i, 64:96] = hot_encode_card(last_trick[0])
        x[0, i, 96:128] = hot_encode_card(last_trick[1])
        x[0, i, 128:160] = hot_encode_card(last_trick[2])
        x[0, i, 160:192] = hot_encode_card(last_trick[3])
        
        x[0, i, 192:224] = hot_encode_card(cards_in[0])
        x[0, i, 224:256] = hot_encode_card(cards_in[1])
        x[0, i, 256:288] = hot_encode_card(cards_in[2])
        
        me_bin[label_card_ix] -= 1
        
    return x, y

def binary_data_dummy(deal_str, outcome_str, play_str):
    x = np.zeros((1, 11, 298), np.float16)
    y = np.zeros((1, 11, 32), np.float16)
    
    hands = list(map(lambda hand_str: list(map(list, hand_str.split('.'))), deal_str[2:].split()))
    d_meta = DealMeta.from_str(outcome_str)
    declarer_i = declarer_index(hands, d_meta, play_str)
    me_i = (declarer_i + 2) % 4

    declarer_bin = binary_hand(hands[declarer_i])
    me_bin = binary_hand(hands[me_i])
    
    _, on_leads, last_tricks, cards_ins, card_outs = get_play_labels(play_str, d_meta.strain, 1)
    
    decl_played_cards = set(['>>'])
    
    for i, (on_lead, last_trick, cards_in, card_out) in enumerate(zip(on_leads, last_tricks, cards_ins, card_outs)):
        if i > 10:
            break
        label_card_ix = get_card_index(card_out)
        y[0, i, label_card_ix] = 1
        x[0, i, 292] = d_meta.level
        if d_meta.strain == 'N':
            x[0, i, 293] = 1
        else:
            x[0, i, 294 + suit_index_lookup[d_meta.strain]] = 1
        
        x[0, i, 288 + on_lead] = 1
        
        last_trick_decl_card = last_trick[3]
        if last_trick_decl_card not in decl_played_cards:
            declarer_bin[get_card_index(last_trick_decl_card)] -= 1
            decl_played_cards.add(last_trick_decl_card)
        
        if cards_in[1] not in decl_played_cards:
            declarer_bin[get_card_index(cards_in[1])] -= 1
            decl_played_cards.add(cards_in[1])
        
        x[0, i, 32:64] = declarer_bin
        x[0, i, 0:32] = me_bin
        
        x[0, i, 64:96] = hot_encode_card(last_trick[0])
        x[0, i, 96:128] = hot_encode_card(last_trick[1])
        x[0, i, 128:160] = hot_encode_card(last_trick[2])
        x[0, i, 160:192] = hot_encode_card(last_trick[3])
        
        x[0, i, 192:224] = hot_encode_card(cards_in[0])
        x[0, i, 224:256] = hot_encode_card(cards_in[1])
        x[0, i, 256:288] = hot_encode_card(cards_in[2])
        
        me_bin[label_card_ix] -= 1
        
    return x, y

def binary_data_lefty(deal_str, outcome_str, play_str):
    x = np.zeros((1, 11, 298), np.float16)
    y = np.zeros((1, 11, 32), np.float16)
    
    hands = list(map(lambda hand_str: list(map(list, hand_str.split('.'))), deal_str[2:].split()))
    d_meta = DealMeta.from_str(outcome_str)
    declarer_i = declarer_index(hands, d_meta, play_str)
    dummy_i = (declarer_i + 2) % 4
    me_i = (declarer_i + 1) % 4

    dummy_bin = binary_hand(hands[dummy_i])
    me_bin = binary_hand(hands[me_i])
    
    _, on_leads, last_tricks, cards_ins, card_outs = get_play_labels(play_str, d_meta.strain, 0)
    
    dummy_played_cards = set(['>>'])
    
    for i, (on_lead, last_trick, cards_in, card_out) in enumerate(zip(on_leads, last_tricks, cards_ins, card_outs)):
        if i > 10:
            break
        label_card_ix = get_card_index(card_out)
        y[0, i, label_card_ix] = 1
        x[0, i, 292] = d_meta.level
        if d_meta.strain == 'N':
            x[0, i, 293] = 1
        else:
            x[0, i, 294 + suit_index_lookup[d_meta.strain]] = 1
        
        x[0, i, 288 + on_lead] = 1
        
        last_trick_dummy_card = last_trick[1]
        if last_trick_dummy_card not in dummy_played_cards:
            dummy_bin[get_card_index(last_trick_dummy_card)] -= 1
            dummy_played_cards.add(last_trick_dummy_card)
        
        if cards_in[0] not in dummy_played_cards:
            dummy_bin[get_card_index(cards_in[0])] -= 1
            dummy_played_cards.add(cards_in[0])
        
        x[0, i, 32:64] = dummy_bin
        x[0, i, 0:32] = me_bin
        
        x[0, i, 64:96] = hot_encode_card(last_trick[0])
        x[0, i, 96:128] = hot_encode_card(last_trick[1])
        x[0, i, 128:160] = hot_encode_card(last_trick[2])
        x[0, i, 160:192] = hot_encode_card(last_trick[3])
        
        x[0, i, 192:224] = hot_encode_card(cards_in[0])
        x[0, i, 224:256] = hot_encode_card(cards_in[1])
        x[0, i, 256:288] = hot_encode_card(cards_in[2])
        
        me_bin[label_card_ix] -= 1
        
    return x, y

def handle_declarer_suit(out_dir, binary_data_decl, data_it_suit, n2):
    X = np.zeros((n2, 11, 298), np.float16)
    Y = np.zeros((n2, 11, 32), np.float16)

    i = 0
    for j, (deal_str, outcome_str, play_str) in enumerate(data_it_suit):
        d_meta = DealMeta.from_str(outcome_str)
        if d_meta.strain != "N":
            if (i != 0) and i % 10000 == 0:
                print(f"\rProgress: {i}/{n2} steps", end="")
            x_i, y_i = binary_data_decl(deal_str, outcome_str, play_str)

            X[i, :, :] = x_i
            Y[i, :, :] = y_i
            i += 1

    np.save(os.path.join(out_dir, 'X_decl_suit.npy'), X)
    np.save(os.path.join(out_dir, 'Y_decl_suit.npy'), Y)
    print()

def handle_declarer_nt(out_dir, binary_data_decl, data_it_nt, n1):
    X = np.zeros((n1, 11, 298), np.float16)
    Y = np.zeros((n1, 11, 32), np.float16)

    i = 0
    for j, (deal_str, outcome_str, play_str) in enumerate(data_it_nt):
        d_meta = DealMeta.from_str(outcome_str)
        if d_meta.strain == "N":
            if (i != 0) and i % 10000 == 0:
                print(f"\rProgress: {i}/{n1} steps", end="")
            x_i, y_i = binary_data_decl(deal_str, outcome_str, play_str)
            X[i, :, :] = x_i
            Y[i, :, :] = y_i
            i += 1

    np.save(os.path.join(out_dir, 'X_decl_nt.npy'), X)
    np.save(os.path.join(out_dir, 'Y_decl_nt.npy'), Y)
    print()

def handle_dummy_nt(out_dir, binary_data_dummy, data_it_nt, n1):
    X = np.zeros((n1, 11, 298), np.float16)
    Y = np.zeros((n1, 11, 32), np.float16)

    i = 0
    for j, (deal_str, outcome_str, play_str) in enumerate(data_it_nt):
        d_meta = DealMeta.from_str(outcome_str)
        if d_meta.strain == "N":
            if (i != 0) and i % 10000 == 0:
                print(f"\rProgress: {i}/{n1} steps", end="")
            x_i, y_i = binary_data_dummy(deal_str, outcome_str, play_str)
            X[i, :, :] = x_i
            Y[i, :, :] = y_i
            i += 1

    np.save(os.path.join(out_dir, 'X_dummy_nt.npy'), X)
    np.save(os.path.join(out_dir, 'Y_dummy_nt.npy'), Y)
    print()

def handle_dummy_suit(out_dir, binary_data_dummy, data_it_suit, n2):
    X = np.zeros((n2, 11, 298), np.float16)
    Y = np.zeros((n2, 11, 32), np.float16)

    i = 0
    for j, (deal_str, outcome_str, play_str) in enumerate(data_it_suit):
        d_meta = DealMeta.from_str(outcome_str)
        if d_meta.strain != "N":
            if (i != 0) and i % 10000 == 0:
                print(f"\rProgress: {i}/{n2} steps", end="")
            x_i, y_i = binary_data_dummy(deal_str, outcome_str, play_str)

            X[i, :, :] = x_i
            Y[i, :, :] = y_i
            i += 1

    np.save(os.path.join(out_dir, 'X_dummy_suit.npy'), X)
    np.save(os.path.join(out_dir, 'Y_dummy_suit.npy'), Y)
    print()

def handle_lefty_nt(out_dir, binary_data_lefty, data_it_nt, n1):
    X = np.zeros((n1, 11, 298), np.float16)
    Y = np.zeros((n1, 11, 32), np.float16)

    i = 0
    for j, (deal_str, outcome_str, play_str) in enumerate(data_it_nt):
        d_meta = DealMeta.from_str(outcome_str)
        if d_meta.strain == "N":
            if (i != 0) and i % 10000 == 0:
                print(f"\rProgress: {i}/{n1} steps", end="")
            x_i, y_i = binary_data_lefty(deal_str, outcome_str, play_str)
            X[i, :, :] = x_i
            Y[i, :, :] = y_i
            i += 1

    np.save(os.path.join(out_dir, 'X_lefty_nt.npy'), X)
    np.save(os.path.join(out_dir, 'Y_lefty_nt.npy'), Y)
    print()

def handle_lefty_suit(out_dir, binary_data_lefty, data_it_suit, n2):
    X = np.zeros((n2, 11, 298), np.float16)
    Y = np.zeros((n2, 11, 32), np.float16)

    i = 0
    for j, (deal_str, outcome_str, play_str) in enumerate(data_it_suit):
        d_meta = DealMeta.from_str(outcome_str)
        if d_meta.strain != "N":
            if (i != 0) and i % 10000 == 0:
                print(f"\rProgress: {i}/{n2} steps", end="")
            x_i, y_i = binary_data_lefty(deal_str, outcome_str, play_str)

            X[i, :, :] = x_i
            Y[i, :, :] = y_i
            i += 1

    np.save(os.path.join(out_dir, 'X_lefty_suit.npy'), X)
    np.save(os.path.join(out_dir, 'Y_lefty_suit.npy'), Y)
    print()

def handle_righty_nt(out_dir, binary_data_righty, data_it_nt, n1):
    X = np.zeros((n1, 11, 298), np.float16)
    Y = np.zeros((n1, 11, 32), np.float16)

    i = 0
    for j, (deal_str, outcome_str, play_str) in enumerate(data_it_nt):
        d_meta = DealMeta.from_str(outcome_str)
        if d_meta.strain == "N":
            if (i != 0) and i % 10000 == 0:
                print(f"\rProgress: {i}/{n1} steps", end="")
            x_i, y_i = binary_data_righty(deal_str, outcome_str, play_str)
            X[i, :, :] = x_i
            Y[i, :, :] = y_i
            i += 1

    np.save(os.path.join(out_dir, 'X_righty_nt.npy'), X)
    np.save(os.path.join(out_dir, 'Y_righty_nt.npy'), Y)
    print()    

def handle_righty_suit(out_dir, binary_data_righty, data_it_suit, n2):
    X = np.zeros((n2, 11, 298), np.float16)
    Y = np.zeros((n2, 11, 32), np.float16)

    i = 0
    for j, (deal_str, outcome_str, play_str) in enumerate(data_it_suit):
        d_meta = DealMeta.from_str(outcome_str)
        if d_meta.strain != "N":
            if (i != 0) and i % 10000 == 0:
                print(f"\rProgress: {i}/{n2} steps", end="")
            x_i, y_i = binary_data_righty(deal_str, outcome_str, play_str)

            X[i, :, :] = x_i
            Y[i, :, :] = y_i
            i += 1

    np.save(os.path.join(out_dir, 'X_righty_suit.npy'), X)
    np.save(os.path.join(out_dir, 'Y_righty_suit.npy'), Y)
    print()

HANDLERS = {
    'decl_nt': (handle_declarer_nt, binary_data_decl), 'dummy_nt': (handle_dummy_nt, binary_data_dummy),
    'lefty_nt': (handle_lefty_nt, binary_data_lefty), 'righty_nt': (handle_righty_nt, binary_data_righty),
    'decl_suit': (handle_declarer_suit, binary_data_decl), 'dummy_suit': (handle_dummy_suit, binary_data_dummy),
    'lefty_suit': (handle_lefty_suit, binary_data_lefty), 'righty_suit': (handle_righty_suit, binary_data_righty),
}

def open_data(inputs):
    # Records whose play does not fit the hands (0.05% of the WBC file) are left out
    return (record for record in play_data_iterator(itertools.chain(*[open(path) for path in inputs]))
            if play_fits_hands(*record))

def run_handler(task):
    # One seat and strain; reads the input files itself, so it can run in its own process
    name, inputs, out_dir, n = task
    handler, binary_data = HANDLERS[name]
    handler(out_dir, binary_data, open_data(inputs), n)
    return name

if __name__ == '__main__':
    import sys
    import multiprocessing

    # python binary_keras_suit_nt.py [input.txt ...] [out_dir=./binary] [workers=4]
    # Without input files: ../../data/WBC/play.txt. Each of the eight arrays is built in its
    # own process, workers at a time; a suit array of 1.2 million deals takes 8 GB while it
    # is built, so four at a time need about 32 GB. workers=1 does them one after the other.
    options = dict(arg.split('=', 1) for arg in sys.argv[1:] if '=' in arg)
    inputs = [arg for arg in sys.argv[1:] if '=' not in arg] or ['../../data/WBC/play.txt']
    out_dir = options.get('out_dir', './binary')
    workers = int(options.get('workers', 4))
    os.makedirs(out_dir, exist_ok=True)

    # Using data from Jack: pass ../../data/jack/BW5C_N.txt ../../data/jack/BW5C_S.txt
    # ../../data/jack/JOS_N.txt ../../data/jack/JOS_S.txt out_dir=./binary_jack

    print("Counting deals")
    n1 = 0
    n2 = 0
    for i, (deal_str, outcome_str, play_str) in enumerate(open_data(inputs)):
        if (i != 0) and i % 100000 == 0:
            print(f"\rProgress: {i} deals", end="")
        d_meta = DealMeta.from_str(outcome_str)
        if d_meta.strain == "N":
            n1 += 1
        else:
            n2 += 1
    print()
    print("{0} total deals".format(n1+n2))
    print(f"Processing {n1} deals for NT and {n2} deals for suit")

    tasks = [(name, inputs, out_dir, n1 if name.endswith('_nt') else n2) for name in HANDLERS]
    if workers > 1:
        with multiprocessing.Pool(workers, maxtasksperchild=1) as pool:
            for name in pool.imap_unordered(run_handler, tasks):
                print()
                print("Done:", name)
    else:
        for task in tasks:
            print(task[0])
            run_handler(task)
