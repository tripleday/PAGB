import secrets
import datetime

from helpfunctions import concat, generate_two_large_safe_primes, hash_to_prime, hash_to_length, bezoute_coefficients,\
    mul_inv, shamir_trick

RSA_KEY_SIZE = 256  # RSA key size for 128 bits of security (modulu size)
RSA_PRIME_SIZE = int(RSA_KEY_SIZE / 2)
ACCUMULATED_PRIME_SIZE = 128  # taken from: LLX, "Universal accumulators with efficient nonmembership proofs", construction 1


def setup():
    # draw strong primes p,q
    p, q = generate_two_large_safe_primes(RSA_PRIME_SIZE)
    n = p*q
    # draw random number within range of [0,n-1]
    # A0 = secrets.randbelow(n)E
    A0 = 65537
    # print(p)
    # print(q)
    # print(n)
    return n, A0, dict()


def add(A, S, x, n):
    if x in S.keys():
        return A
    else:
        hash_prime, nonce = hash_to_prime(x, ACCUMULATED_PRIME_SIZE)
        A = pow(A, hash_prime, n)
        S[x] = nonce
        return A


def batch_add_test(A_pre_add, x_list, n, p, q):
    product = 1
    test = 1
    A_post_add = A_pre_add
    for x in x_list:
        # if x not in S.keys():
            hash_prime, nonce = hash_to_prime(x, ACCUMULATED_PRIME_SIZE)
            print(hash_prime)
            # S[x] = nonce
            product *= hash_prime
            test = (test * hash_prime) % ((p-1)*(q-1))
            A_post_add = pow(A_post_add, hash_prime, n)

    print(A_post_add)
    A_post = pow(A_pre_add, product, n)
    print(A_post)
    test_post = pow(A_pre_add, test, n)
    print(test_post)
    return A_post_add
    # return test_post

def batch_add(A_pre_add, S, x_list, n):
    A_post_add = A_pre_add
    for x in x_list:
        if x not in S.keys():
            # print(x)
            hash_prime, nonce = hash_to_prime(x, ACCUMULATED_PRIME_SIZE)
            # print(hash_prime)
            S[x] = nonce
            A_post_add = pow(A_post_add, hash_prime, n)
        else:
            print(x)
    return A_post_add


def prove_membership(A0, S, x, n):
    if x not in S.keys():
        return None
    else:
        # A = A0      
        for element in S.keys():
            if element != x:
                nonce = S[element]
                product *= hash_to_prime(element, ACCUMULATED_PRIME_SIZE, nonce)[0]
                # A = pow(A, hash_to_prime(element, ACCUMULATED_PRIME_SIZE, nonce)[0], n)

        A = pow(A0, product, n)
        return A


def prove_non_membership(A0, S, x, x_nonce, n):
    if x in S.keys():
        return None
    else:
        product = 1
        for element in S.keys():
            nonce = S[element]
            product *= hash_to_prime(element, ACCUMULATED_PRIME_SIZE, nonce)[0]
    prime = hash_to_prime(x, ACCUMULATED_PRIME_SIZE, x_nonce)[0]
    # print(prime)
    # print(product)
    a, b = bezoute_coefficients(prime, product)
    # print(a)
    # print(b)
    # print(pow(A0, a, n))
    # if a < 0:
    #     positive_a = -a
    #     inverse_A0 = mul_inv(A0, n)
    #     d = pow(inverse_A0, positive_a, n)
    #     print(d)
    # else:
    #     d = pow(A0, a, n)
    d = pow(A0, a, n)
    return d, b


def verify_non_membership(A0, A_final, d, b, x, x_nonce, n):
    prime = hash_to_prime(x, ACCUMULATED_PRIME_SIZE, x_nonce)[0]
    # if b < 0:
    #     positive_b = -b
    #     inverse_A_final = mul_inv(A_final, n)
    #     second_power = pow(inverse_A_final, positive_b, n)
    # else:
    #     second_power = pow(A_final, b, n)
    second_power = pow(A_final, b, n)
    return (pow(d, prime, n) * second_power) % n == A0


# NI-PoE: non-interactive version of section 3.1 in BBF18 (PoE).
# Receives:
#   u - the accumulator value before add
#   x - the (prime) element which was added to the accumulator
#   w - the accumulator after the addition of x
#   n - the modulu
# Returns:
#   Q, x - the NIPoE

def prove_exponentiation_test(u, x, w, n):
    l, nonce = hash_to_prime(concat(x, u, w))  # Fiat-Shamir instead of interactive challenge
    q = x // l
    Q = pow(u, q, n)
    return Q

# Verify NI-PoE
# The verifier has to reproduce l himself.
def verify_exponentiation_test(Q, u, x, w, n):
    # x = hash_to_prime(x=x)[0]
    return __verify_exponentiation_test(Q, u, x, w, n)

# helper function, does not do hash_to_prime on x
def __verify_exponentiation_test(Q, u, x, w, n):
    # start_time = datetime.datetime.now() 
    l = hash_to_prime(x=(concat(x, u, w)))[0]               
    # end_time = datetime.datetime.now()
    # print((end_time - start_time))

    # start_time = datetime.datetime.now()
    r = x % l        
    # end_time = datetime.datetime.now()
    # print((end_time - start_time))
    # check (Q^l)(u^r) == w
    return pow(Q, l, n) * pow(u, r, n) % n == w



# NI-PoKE2: non-interactive version of section 3.2 in BBF18 (PoKE2).
# Receives:
#   u - the accumulator value before add
#   x - the (prime) element which was added to the accumulator
#   w - the accumulator after the addition of x
#   n - the modulu
# Returns:

def prove_knowledge_exponent(x, u, w, n):
    g = pow(65537, hash_to_length(concat(u, w)), n)
    z = pow(g, x, n)
    l, nonce = hash_to_prime(concat(u, w, z))  # Fiat-Shamir instead of interactive challenge
    alpha = hash_to_length(concat(u, w, z, l))
    q = x // l
    r = x % l
    Q = pow(u*pow(g,alpha,n), q, n)
    return z, Q, r

def verify_knowledge_exponent(u, w, z, Q, r, n):
    g = pow(65537, hash_to_length(concat(u, w)), n)
    l, nonce = hash_to_prime(concat(u, w, z))
    alpha = hash_to_length(concat(u, w, z, l))
    # print((pow(Q, l, n) * pow(u*pow(g,alpha,n), r, n)) % n)
    # print((w * pow(z, alpha, n)) % n )
    return (pow(Q, l, n) * pow(u*pow(g,alpha,n), r, n)) % n == (w * pow(z, alpha, n)) % n 