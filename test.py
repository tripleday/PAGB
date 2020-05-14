import secrets
import math
import hashlib
from helpfunctions import hash_to_prime, is_prime, shamir_trick, concat, bezoute_coefficients, mul_inv
from main import setup, add, prove_membership, \
        batch_add, batch_add_test,\
        prove_non_membership, verify_non_membership,\
        prove_exponentiation_test, verify_exponentiation_test, prove_knowledge_exponent, verify_knowledge_exponent
from unittest import TestCase
import unittest
import datetime
import numpy as np
import pandas as pd
import pickle
import sys
import json
import random


def create_list(size):
        res = []
        for i in range(size):
                x = secrets.randbelow(pow(2, 256))
                res.append(x)
        return res


class AccumulatorTest(TestCase):
        def test_hash_to_prime(self):
                x = secrets.randbelow(pow(2, 256))
                #print(x)
                h, nonce = hash_to_prime(x, 128)
                #print(h)
                #print(nonce)
                self.assertTrue(is_prime(h))
                self.assertTrue(h, math.log2(h) < 128)



        def test_batch_add_euler(self):
                p = 252533614457563255817176556954479732787
                q = 144382690536755041477176940054403505719
                n = 36461482706354564422592875042006590908268153693683612285024099145347146308853
                # draw random number within range of [0,n-1]
                A0 = 65537
                # S = dict()
              

                elements_list = []
                counter = 0
                start_time = datetime.datetime.now()
                for line in open("CA-CondMat.txt"):   
                        # print(line)
                        if line.split(' ', 1)[0] == '#':
                                # print(1)
                                continue
                        # print(line.split('\t', 1))
                        if counter >= 3:
                                break
                        element = line.split('\t', 1)[0]+'to'+line.split('\t', 1)[1]
                        element_int = int(hashlib.sha256(element.encode("utf-8")).hexdigest(), 16)
                        # print(element)
                        elements_list.append(element_int)
                        counter = counter + 1
                # print(counter)


                A_post_add = batch_add_test(A0, elements_list, n, p, q)

                end_time = datetime.datetime.now()
                print((end_time - start_time))


        def test_ca(self):
                p = 252533614457563255817176556954479732787
                q = 144382690536755041477176940054403505719
                n = 36461482706354564422592875042006590908268153693683612285024099145347146308853
                # draw random number within range of [0,n-1]
                A0 = 65537
                # S = dict()
              

                elements_list = []
                counter = 0
                for line in open("CA-CondMat.txt"):  
                        if counter >= 100:
                                break 
                        if line.split(' ', 1)[0] == '#':
                                #print(1)
                                continue
                        #print(line.split('\t', 1))
                        element = line.split('\t', 1)[0]+'to'+line.split('\t', 1)[1]
                        element_int = int(hashlib.sha256(element.encode("utf-8")).hexdigest(), 16)
                        #print(element)
                        elements_list.append(element_int)
                        counter = counter + 1
                # print(counter)


                A_post_add = batch_add_test(A0, elements_list, n, p ,q)
                # print(len(S))
                # self.assertEqual(len(S), 10)

                # nonces_list = list(map(lambda e: hash_to_prime(e)[1], elements_list))
                # is_valid = batch_verify_membership_with_NIPoE(nipoe[0], nipoe[1], A0, elements_list, nonces_list, A_post_add, n)
                # self.assertTrue(is_valid)


        def test_eff(self):
                # p = 252533614457563255817176556954479732787
                # q = 144382690536755041477176940054403505719
                n = 36461482706354564422592875042006590908268153693683612285024099145347146308853
                # draw random number within range of [0,n-1]
                A0 = 65537
                S = dict()
              

                start_time = datetime.datetime.now()
                elements_list = []
                counter = 0
                product = 1
                for line in open("CA-CondMat.txt"):   
                        if counter >= 10000:
                                break
                        if line.split(' ', 1)[0] == '#':
                                #print(1)
                                continue
                        #print(line.split('\t', 1))
                        element = line.split('\t', 1)[0]+'to'+line.split('\t', 1)[1]
                        element_int = int(hashlib.sha256(element.encode("utf-8")).hexdigest(), 16)
                        #print(element)
                        elements_list.append(element_int)
                        hash_prime, nonce = hash_to_prime(element_int)
                        counter = counter + 1
                        S[element_int] = nonce
                        product *= hash_prime
                print(product)                        
                A_post_add = pow(A0, product, n)
                print(A_post_add) 

                end_time = datetime.datetime.now()
                print((end_time - start_time))


        def test_ca_non_membership(self):
                # p = 252533614457563255817176556954479732787
                # q = 144382690536755041477176940054403505719
                n = 36461482706354564422592875042006590908268153693683612285024099145347146308853
                # draw random number within range of [0,n-1]
                A0 = 65537
                S = dict()
              

                elements_list = []
                counter = 0
                for line in open("CA-CondMat.txt"):   
                        if counter >= 10:
                                break
                        if line.split(' ', 1)[0] == '#':
                                #print(1)
                                continue
                        #print(line.split('\t', 1))
                        element = line.split('\t', 1)[0]+'to'+line.split('\t', 1)[1]
                        element_int = int(hashlib.sha256(element.encode("utf-8")).hexdigest(), 16)
                        #print(element)
                        elements_list.append(element_int)
                        counter = counter + 1
                # print(counter)


                A_post_add = batch_add(A0, S, elements_list, n)
                # print(len(S))

                start_time = datetime.datetime.now()
 
                proof = prove_non_membership(A0, S, elements_list[0], S[elements_list[0]], n)
                # self.assertIsNone(proof)

                x = 7
                prime, x_nonce = hash_to_prime(x)
                print(prime)
                proof = prove_non_membership(A0, S, x, x_nonce, n)
                is_valid = verify_non_membership(A0, A_post_add, proof[0], proof[1], x, x_nonce, n)
                # self.assertTrue(is_valid)
                print(is_valid)


                end_time = datetime.datetime.now()

                print((end_time - start_time))


        def test_ca_poe(self):
                n = 36461482706354564422592875042006590908268153693683612285024099145347146308853
                A0 = 65537
                S = dict()
              

                elements_list = []
                counter = 0
                for line in open("CA-CondMat.txt"):   
                        if counter >= 10:
                                break
                        if line.split(' ', 1)[0] == '#':
                                #print(1)
                                continue
                        #print(line.split('\t', 1))
                        element = line.split('\t', 1)[0]+'to'+line.split('\t', 1)[1]
                        element_int = int(hashlib.sha256(element.encode("utf-8")).hexdigest(), 16)
                        elements_list.append(element_int)
                        counter = counter + 1
                # print(counter)
                A_post_add = batch_add(A0, S, elements_list, n)

                print(A0)
                first = 1
                for e in elements_list:
                        first *= hash_to_prime(x=e, nonce=S[e])[0]
                print(first)
                print(pow(A0, first, n))
                print(A_post_add)

                start_time = datetime.datetime.now()
 
                proof = prove_exponentiation_test(A0, first, A_post_add, n)
                is_valid = verify_exponentiation_test(proof, A0, first, A_post_add, n)
                # self.assertTrue(is_valid)
                print(is_valid)

                end_time = datetime.datetime.now()
                print((end_time - start_time))


        def test_ca_poke2(self):
                n = 36461482706354564422592875042006590908268153693683612285024099145347146308853
                A0 = 65537
                S = dict()
              

                elements_list = []
                counter = 0
                for line in open("CA-CondMat.txt"):   
                        if counter >= 100:
                                break
                        if line.split(' ', 1)[0] == '#':
                                #print(1)
                                continue
                        #print(line.split('\t', 1))
                        element = line.split('\t', 1)[0]+'to'+line.split('\t', 1)[1]
                        element_int = int(hashlib.sha256(element.encode("utf-8")).hexdigest(), 16)
                        elements_list.append(element_int)
                        counter = counter + 1
                # print(counter)
                A_post_add = batch_add(A0, S, elements_list, n)

                print(A0)
                first = 1
                for e in elements_list:
                        first *= hash_to_prime(x=e, nonce=S[e])[0]
                print(first)
                print(first.bit_length())
                start_time = datetime.datetime.now()
                print(pow(A0, first, n)==A_post_add)
                end_time = datetime.datetime.now()
                print((end_time - start_time))
                print(A_post_add)

 
                proof = prove_knowledge_exponent(first, A0, A_post_add, n)

                start_time = datetime.datetime.now()
                is_valid = verify_knowledge_exponent(A0, A_post_add, proof[0], proof[1], proof[2], n)
                # self.assertTrue(is_valid)
                print(is_valid)
                end_time = datetime.datetime.now()
                print((end_time - start_time))


        def test_batch_delete(self):
               # p = 252533614457563255817176556954479732787
                # q = 144382690536755041477176940054403505719
                n = 36461482706354564422592875042006590908268153693683612285024099145347146308853
                # draw random number within range of [0,n-1]
                A0 = 65537
                S = dict()
              

                elements_list = []
                counter = 0
                for line in open("CA-CondMat.txt"):   
                        if counter >= 10:
                                break
                        if line.split(' ', 1)[0] == '#':
                                #print(1)
                                continue
                        #print(line.split('\t', 1))
                        element = line.split('\t', 1)[0]+'to'+line.split('\t', 1)[1]
                        element_int = int(hashlib.sha256(element.encode("utf-8")).hexdigest(), 16)
                        #print(element)
                        elements_list.append(element_int)
                        counter = counter + 1
                # print(counter)


                A_post_add = batch_add(A0, S, elements_list, n)

                elements_to_delete_list = [elements_list[0], elements_list[2], elements_list[4]]
                nonces_list = list(map(lambda e: hash_to_prime(e)[1], elements_to_delete_list))

                proofs = list(map(lambda x: prove_membership(A0, S, x, n), elements_to_delete_list))

                A_post_delete, nipoe = batch_delete_using_membership_proofs(A_post_add, S, elements_to_delete_list, proofs, n)

                # is_valid = batch_verify_membership_with_NIPoE(nipoe[0], nipoe[1], A_post_delete, elements_to_delete_list, nonces_list, A_pre_delete, n)
                # self.assertTrue(is_valid)


        def test(self):
                # n, A0, S = setup()
                # print(is_prime((p-1)//2))
                # print(is_prime((q-1)//2))

                n = 36461482706354564422592875042006590908268153693683612285024099145347146308853
                A0 = 65537
                f = open('wiki_1m_prime.pk','rb')  
                primes_list = pickle.load(f)


                for total in [100000]:
                        print(total)
                        primes_list = primes_list[:total]

                        start_time = datetime.datetime.now() 
                        A_final = A0
                        for e in primes_list:        
                                A_final = pow(A_final, e, n) 
                        end_time = datetime.datetime.now()     
                        print((end_time - start_time)) 


                        start_time = datetime.datetime.now() 
                        product = listProduct(primes_list)
                        acc = pow(A0,product,n)
                        end_time = datetime.datetime.now()     
                        print((end_time - start_time)) 


                        print(A_final==acc)

                # for i in range(73):
                #         if (pow(2,i,91)) == 1:
                #                 print(i)

                # start_time = datetime.datetime.now()
                # t = 2**2560000
                # end_time = datetime.datetime.now()
                # print((end_time - start_time))

                # start_time = datetime.datetime.now()
                # t//256;
                # end_time = datetime.datetime.now()
                # print((end_time - start_time))


        def test_ca_completeness(self):
                p = 252533614457563255817176556954479732787
                q = 144382690536755041477176940054403505719
                n = 36461482706354564422592875042006590908268153693683612285024099145347146308853
                # draw random number within range of [0,n-1]
                A0 = 65537
                S = dict()
              

                num = [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000, 150000]
                for total in num:
                        print(total)
                        C = dict()
                        elements_list = []
                        counter = 0
                        ids = set()
                        edges = set()
                        # edgesto = set()
                        for line in open("CA-CondMat.txt"):  
                                if counter >= total:
                                        break 
                                if line.split(' ', 1)[0] == '#':
                                        #print(1)
                                        continue
                                # print(line.split('\t', 1))
                                start = line.split('\t')[0]
                                end = line.split('\t')[1].split('\n')[0]

                                ids.update({start, end})
                                edges.update({start + ',' + end,})
                                # edgesto.update(start + 'to' + end)

                                if (start+'out') in C:
                                        C[start+'out'].append(end);
                                else:
                                        C[start+'out'] = [end];                    
                                if (end+'in') in C:
                                        C[end+'in'].append(start);
                                else:
                                        C[end+'in'] = [start];
                                counter = counter + 1
                        print(counter)

                        for e in set.union(ids, edges):
                                # print(element)
                                element_int = int(hashlib.sha256(e.encode("utf-8")).hexdigest(), 16)
                                elements_list.append(element_int)

                        for k in C.keys():
                                res = ""
                                for i in range(len(C[k])):
                                        if (i>=1):
                                                res += ','
                                        res += str(C[k][i])
                                element = k+':'+res+';'
                                element_int = int(hashlib.sha256(element.encode("utf-8")).hexdigest(), 16)
                                # print(element)
                                elements_list.append(element_int)
                        #         counter = counter + 1
                        print(len(ids))
                        print(len(edges))
                        print(len(elements_list)-len(ids)-len(edges))
                        print(len(elements_list))

                        # # print(len(C))
                        print('\n')

                # start_time = datetime.datetime.now()
                # A_post_add = batch_add(A0, S, elements_list, n)
                # end_time = datetime.datetime.now()
                # print((end_time - start_time))
                # print(len(S))


        def test_cage_completeness(self):
                p = 252533614457563255817176556954479732787
                q = 144382690536755041477176940054403505719
                n = 36461482706354564422592875042006590908268153693683612285024099145347146308853
                # draw random number within range of [0,n-1]
                A0 = 65537
                S = dict()
              

                num = [100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000, 550000, 600000]
                for total in num:
                        print(total)
                        C = dict()
                        elements_list = []
                        counter = 0
                        ids = set()
                        edges = set()
                        edgesto = set()
                        edgestoweight = set()
                        for line in open("cage15.mtx"):  
                                if line[0] == '%':
                                        continue
                                if counter == 0:
                                        counter = counter + 1
                                        continue
                                if counter >= total+1:
                                        break
                                # print(line)
                                start = line.split(' ', 2)[0]
                                end = line.split(' ', 2)[1]
                                weight = line.split(' ', 2)[2].split('\n')[0]


                                ids.update({start, end})
                                edges.update({start + ',' + end,})
                                # edgesto.update({start + 'to' + end,})
                                edgestoweight.update({start + ',' + end + ':' + weight,})
                                # element = start + 'to' + end + ':' + weight + ';'
                                # print(element)

                                if (start+'out') in C:
                                        C[start+'out'].append(end);
                                else:
                                        C[start+'out'] = [end];                    
                                if (end+'in') in C:
                                        C[end+'in'].append(start);
                                else:
                                        C[end+'in'] = [start];

                                counter = counter + 1
                        print(counter)

                        for e in set.union(ids, edges, edgesto, edgestoweight):
                                # print(element)
                                element_int = int(hashlib.sha256(e.encode("utf-8")).hexdigest(), 16)
                                elements_list.append(element_int)

                        for k in C.keys():                   
                                res = ""
                                for i in range(len(C[k])):
                                        if (i>=1):
                                                res += ','
                                        res += str(C[k][i])
                                element = k+':'+res+';'
                                element_int = int(hashlib.sha256(element.encode("utf-8")).hexdigest(), 16)
                                # print(element)
                                elements_list.append(element_int)
                                # counter = counter + 1

                        print(len(ids))
                        print(len(edgestoweight))
                        print(len(elements_list)-len(ids)-len(edgestoweight))
                        print(len(elements_list))
                        # print(len(C))
                        print('\n')

                # start_time = datetime.datetime.now()
                # A_post_add = batch_add(A0, S, elements_list, n)
                # end_time = datetime.datetime.now()
                # print((end_time - start_time))
                # print(len(S))


        def test_wiki_completeness(self):
                p = 252533614457563255817176556954479732787
                q = 144382690536755041477176940054403505719
                n = 36461482706354564422592875042006590908268153693683612285024099145347146308853
                # draw random number within range of [0,n-1]
                A0 = 65537
                S = dict()
                # f = open('wiki.pk','rb')  
                # elements_list = pickle.load(f)
                # print(len(elements_list))


                num = [100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000, 550000, 600000]
                for total in num:
                        print(total)
                        C = dict()
                        O = dict()
                        counter = 0
                        ids = set()
                        edges = set()
                        edgesto = set()
                        edgestoweight = set()
                        elements_list = []
                        for line in open("wiki-talk-temporal.txt"):  
                                if counter >= total:
                                        break
                                # print(line)
                                start = line.split(' ', 2)[0]
                                end = line.split(' ', 2)[1]
                                weight = line.split(' ', 2)[2].split('\n')[0]
                                # element = start + 'to' + end + ':' + weight + ','
                                # print(element)

                                ids.update({start, end})
                                edges.update({start + ',' + end,})
                                # edgesto.update({start + 'to' + end,})
                                # edgestoweight.update({start + 'to' + end + ':' + weight,})

                                if (start + ',' + end) in O:
                                        O[start + ',' + end].append(weight);
                                else:
                                        O[start + ',' + end] = [weight];   


                                if (start+'out') in C:
                                        C[start+'out'].append(end);
                                else:
                                        C[start+'out'] = [end];                    
                                if (end+'in') in C:
                                        C[end+'in'].append(start);
                                else:
                                        C[end+'in'] = [start];

                                # element_int = int(hashlib.sha256(element.encode("utf-8")).hexdigest(), 16)
                                # # print(element)
                                # elements_list.append(element_int)
                                counter = counter + 1
                        print(counter)

                        for k in O.keys():                   
                                res = ""
                                for i in range(len(O[k])):
                                        if (i>=1):
                                                res += ','
                                        res += str(O[k][i])
                                element = k+':'+res+';'
                                edgestoweight.update({element,})
                                # element_int = int(hashlib.sha256(element.encode("utf-8")).hexdigest(), 16)
                                # # print(element)
                                # elements_list.append(element_int)
                                # counter = counter + 1

                        for e in set.union(ids, edges, edgesto, edgestoweight):
                                # print(element)
                                element_int = int(hashlib.sha256(e.encode("utf-8")).hexdigest(), 16)
                                elements_list.append(element_int)


                        for k in C.keys():                   
                                res = ""
                                for i in range(len(C[k])):
                                        if (i>=1):
                                                res += ','
                                        res += str(C[k][i])
                                element = k+':'+res+';'
                                element_int = int(hashlib.sha256(element.encode("utf-8")).hexdigest(), 16)
                                # print(element)
                                elements_list.append(element_int)
                                # counter = counter + 1

                        print(len(ids))
                        # print(len(edges))
                        # print(len(edgesto))
                        print(len(edgestoweight))
                        print(len(elements_list)-len(ids)-len(edgestoweight))
                        print(len(elements_list))
                        # # print(len(C))
                        print('\n')


                # start_time = datetime.datetime.now()
                # A_post_add = batch_add(A0, S, elements_list, n)
                # end_time = datetime.datetime.now()
                # print((end_time - start_time))
                # print(len(S))


        def test_wiki_pickle(self):            
                O = dict()
                C = dict()
                elements_list = []
                counter = 0
                for line in open("wiki-talk-temporal.txt"):  
                        # if counter >= 50000:
                        #         break
                        # print(line)
                        start = line.split(' ', 2)[0]
                        end = line.split(' ', 2)[1]
                        weight = line.split(' ', 2)[2].split('\n')[0]
                        element = start + 'to' + end + ':' + weight + ','
                        # print(element)

                        if (start + 'to' + end) in O:
                                O[start + 'to' + end].append(weight);
                        else:
                                O[start + 'to' + end] = [weight];   


                        if (start+'out') in C:
                                C[start+'out'].append(end);
                        else:
                                C[start+'out'] = [end];                    
                        if (end+'in') in C:
                                C[end+'in'].append(start);
                        else:
                                C[end+'in'] = [start];

                        element_int = int(hashlib.sha256(element.encode("utf-8")).hexdigest(), 16)
                        # if element_int in elements_list:
                        #         print(element)
                        # else:
                        #         elements_list.append(element_int)
                        #         counter = counter + 1
                        elements_list.append(element_int)   
                        counter = counter + 1
                print(counter)

                for k in O.keys():                   
                        res = ""
                        for i in range(len(O[k])):
                                if (i>=1):
                                        res += ','
                                res += str(O[k][i])
                        element = k+':'+res+';'
                        element_int = int(hashlib.sha256(element.encode("utf-8")).hexdigest(), 16)
                        # if element_int in elements_list:
                        #         print(element)
                        # else:
                        #         elements_list.append(element_int)
                        #         counter = counter + 1
                        elements_list.append(element_int)   
                        counter = counter + 1

                for k in C.keys():                   
                        res = ""
                        for i in range(len(C[k])):
                                if (i>=1):
                                        res += ','
                                res += str(C[k][i])
                        element = k+':'+res+';'
                        element_int = int(hashlib.sha256(element.encode("utf-8")).hexdigest(), 16)
                        # if element_int in elements_list:
                        #         print(element)
                        # else:
                        #         elements_list.append(element_int)
                        #         counter = counter + 1
                        elements_list.append(element_int)    
                        counter = counter + 1  

                # print(len(C))
                print(counter)

                print(len(elements_list))
                print(len(set(elements_list)))
                elements_list = list(set(elements_list))
                print(len(elements_list))
                with open('wiki.pk', 'wb') as f:
                        pickle.dump(elements_list, f)


        def test_concept_completeness(self):
                # p = 252533614457563255817176556954479732787
                # q = 144382690536755041477176940054403505719
                # n = 36461482706354564422592875042006590908268153693683612285024099145347146308853
                # # draw random number within range of [0,n-1]
                # A0 = 65537
                # S = dict()

                data = pd.read_csv('chineseconceptnet.csv', delimiter='\t')
                data.columns = ['uri', 'relation', 'start', 'end', 'json']
                weights = data['json'].apply(lambda row: json.loads(row)['weight'])
                datasets = data['json'].apply(lambda row: json.loads(row)['dataset'])
                data.pop('json')
                data.insert(4,'weights',weights)
                data.insert(5,'dataset',datasets)


                # print(data[0])
                # print(len(data))
              

                num = [100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000, 550000, 600000]
                # num = [300000]
                for total in num:
                        print(total)
                        elements_list = []
                        counter = 0
                        ids = set()
                        edges = set()
                        edgesto = set()
                        edgestoweight = set()
                        edgesout = set()
                        edgesin = set()
                        C = dict()
                        for i in range(len(data)):  
                                if counter >= total:
                                        break
                                # print(data.loc[i]['end'])
                                start = data.loc[i]['start']
                                end = data.loc[i]['end']
                                relation = data.loc[i]['relation']
                                weight = data.loc[i]['weights']
                                dataset = data.loc[i]['dataset']


                                ids.update({start, end})
                                edges.update({start + ',' + end,})
                                edgesto.update({start +','+ relation +','+  end,})
                                # if (start +','+  relation +','+  end + ',' + str(weight) in edgestoweight):
                                #         print(counter)
                                #         print(start +','+  relation +','+  end + ',' + str(weight))
                                edgestoweight.update({start +','+  relation +','+  end + ',weight',})
                                edgestoweight.update({start +','+  relation +','+  end + ',weight:' + str(weight),})
                                edgestoweight.update({start +','+  relation +','+  end + ',dataset',})
                                edgestoweight.update({start +','+  relation +','+  end + ',dataset:' + str(dataset),})
                                edgesout.update({start +',out,'+ relation,})
                                edgesin.update({end +',in,'+  relation,})
                                # element = start + 'to' + end + ':' + weight + ';'
                                # print(element)

                                if (start+',out,') in C:
                                        C[start+',out,'+ relation].update({end},);
                                else:
                                        C[start+',out,'+ relation] = set([end]);                    
                                if (end+',in,') in C:
                                        C[end+',in,'+ relation].update({start},);
                                else:
                                        C[end+',in,'+ relation] = set([start]);                
                                if (start + ',' + end) in C:
                                        C[start + ',' + end].update({relation},);
                                else:
                                        C[start + ',' + end] = set([relation]);

                                counter = counter + 1
                        print(counter)

                        for e in set.union(ids, edges, edgesto, edgestoweight, edgesout, edgesin):
                                # print(element)
                                element_int = int(hashlib.sha256(e.encode("utf-8")).hexdigest(), 16)
                                elements_list.append(element_int)

                        for k in C.keys():                   
                                res = ""
                                # t = len(C[k])
                                for i in range(len(C[k])):
                                        if (i>=1):
                                                res += ','
                                        res += str(C[k].pop())
                                element = k+':'+res+';'
                                # if (t>1):
                                #         print(element)
                                element_int = int(hashlib.sha256(element.encode("utf-8")).hexdigest(), 16)
                                # print(element)
                                elements_list.append(element_int)
                                # counter = counter + 1

                        print(len(ids))
                        # print(len(edges))
                        # print(len(edgesto))
                        print(len(edgesto))
                        # print(len(edgesout))
                        # print(len(edgesin))
                        print(len(elements_list)-len(ids)-len(edgesto))
                        print(len(elements_list))
                        # print(len(set(elements_list)))
                        # # print(len(C))
                        print('\n')

                        # # start_time = datetime.datetime.now()
                        # # A_post_add = batch_add(A0, S, elements_list, n)
                        # # end_time = datetime.datetime.now()
                        # # print((end_time - start_time))
                        # # print(len(S))


        def test_prime_prepare(self):
                         
                primes_list = []


                f = open('wiki.pk','rb')  
                elements_list = pickle.load(f)[:1000000]
                print(len(elements_list))
                for e in elements_list:
                        hash_prime, nonce = hash_to_prime(e)
                        primes_list.append(hash_prime)
                print(len(primes_list))

                with open('wiki_1m_prime.pk', 'wb') as f:
                        pickle.dump(primes_list, f)


        def test_original_wit(self):
                # p = 252533614457563255817176556954479732787
                # q = 144382690536755041477176940054403505719
                # n = 36461482706354564422592875042006590908268153693683612285024099145347146308853
                # draw random number within range of [0,n-1]
                # A0 = 65537
                # S = dict()             

                # primes_list = []
                # counter = 0
                # product = 1
                # for line in open("wiki-talk-temporal.txt"):  
                #         if counter >= 1000000:
                #                 break
                #         start = line.split(' ', 2)[0]
                #         end = line.split(' ', 2)[1]
                #         weight = line.split(' ', 2)[2].split('\n')[0]
                #         element = start + 'to' + end + ':' + weight + ','
                #         element_int = int(hashlib.sha256(element.encode("utf-8")).hexdigest(), 16)
                #         #print(element)
                #         hash_prime, nonce = hash_to_prime(element_int)
                #         primes_list.append(hash_prime)
                #         counter = counter + 1
                        # S[element_int] = nonce
                        # product *= hash_prime
                # print(product)
                f = open('wiki_1m_prime.pk','rb')  
                prime_list = pickle.load(f)


                # f = open('wiki.pk','rb')  
                # elements_list = pickle.load(f)[:1000000]
                # print(len(elements_list))
                # for e in elements_list:
                #         hash_prime, nonce = hash_to_prime(element_int)
                #         primes_list.append(hash_prime)
                # print(len(primes_list))

                # with open('wiki_1m_prime.pk', 'wb') as f:
                #         pickle.dump(primes_list, f)

                for total in [10000, 40000, 90000, 160000, 250000, 360000, 490000, 640000, 810000, 1000000]:
                        print(total)
                        product = 1
                        primes_list = prime_list[:total]
                        start_time = datetime.datetime.now() 
                        for e in primes_list:
                                product *= e
                        end_time = datetime.datetime.now()
                        print((end_time - start_time))

                        # print(len(primes_list))
                        # print(sys.getsizeof(primes_list))
                        # print(product.bit_length())
                        # print(sys.getsizeof(product))
                        print('\n')


        def test_ed_wit(self):
                f = open('wiki_1m_prime.pk','rb')  
                primes_list = pickle.load(f)


                for total in [30000, 20000, 10000]:
                        print(total)
                        primes_list = primes_list[:total]
                        lproduct = 1
                        rproduct = 1
                        product_list = []
                        for i in range(total):
                                product_list.append([0]*2)
                        for i in range(total):
                                product_list[i][0] = lproduct
                                product_list[-1-i][1] = rproduct
                                lproduct *= primes_list[i]
                                rproduct *= primes_list[-1-i]

                        start_time = datetime.datetime.now() 
                        product = product_list[5][0] * product_list[5][1]
                        # product2 = product_list[51][0] * product_list[51][1]
                        # print(product1*primes_list[50]==product2*primes_list[51])
                        end_time = datetime.datetime.now()
                        print((end_time - start_time))

                        # print(product_list)
                        print(len(product_list))
                        size = 0
                        for e in product_list:
                                size += sys.getsizeof(e[0]) + sys.getsizeof(e[1])
                        print(size)
                        # print(primes_list)
                        # print(len(primes_list))
                        # print(sys.getsizeof(primes_list))
                        # print(product.bit_length())
                        # print(sys.getsizeof(product))
                        print('\n')


        def test_acc(self):
                n = 36461482706354564422592875042006590908268153693683612285024099145347146308853
                # draw random number within range of [0,n-1]
                A0 = 65537
                f = open('wiki.pk','rb')  
                int_list = pickle.load(f)[:1000000]

                for total in [1000000, 900000, 800000, 700000, 600000, 500000, 400000, 300000, 200000, 100000]:
                        print(total)
                        int_list = int_list[:total]
                        prime_list = []

                        start_time = datetime.datetime.now() 
                        for e in int_list:
                                hash_prime, nonce = hash_to_prime(e)
                                prime_list.append(hash_prime)                  
                        end_time = datetime.datetime.now()
                        print((end_time - start_time))

                        # product = 1
                        start_time = datetime.datetime.now() 
                        for e in prime_list:
                                product *= e    
                        # product = listProduct(prime_list)           
                        end_time = datetime.datetime.now()
                        print((end_time - start_time))

                        start_time = datetime.datetime.now() 
                        A_post_add = pow(A0, product, n)      
                        end_time = datetime.datetime.now()
                        print((end_time - start_time))
                        print(A_post_add) 
                        print('\n')


        def test_prove_verify(self):               
                n = 36461482706354564422592875042006590908268153693683612285024099145347146308853
                A0 = 65537
                f = open('wiki_1m_prime.pk','rb')  
                primes_list = pickle.load(f)
                primes_list += primes_list[:50000]


                for total in [1000000, 900000, 800000, 700000, 600000, 500000, 400000, 300000, 200000, 100000]:
                        print(total)
                        primes_list = primes_list[:total]
                        K1 = math.ceil(pow(total/8,1/4))
                        K2 = 2*K1
                        K3 = 4*K1
                        product_list1 = []
                        product_list2 = []
                        product_list3 = []


                        for i in range(K3):
                                product_list2.append([])
                                product_list3.append([])
                                for j in range(K2):
                                        product_list3[i].append([])
                        for i in range(K3):
                                product1 = 1
                                for j in range(K2):
                                        product2 = 1
                                        for k in range(K1):
                                                product3 = 1
                                                for l in range(K1):
                                                        product3 *= primes_list[i*K1*K1*K2 + j*K1*K1 + k*K1 + l]
                                                product_list3[i][j].append(product3)
                                                product2 *= product3
                                        product_list2[i].append(product2)
                                        product1 *= product2
                                product_list1.append(product1)     

                        product = 1
                        for e in product_list1:
                                product *= e   
                        A_final = pow(A0, product, n) 

                        start_time = datetime.datetime.now() 
                        mwp = 1
                        for i in range(K3-1):
                                mwp *= product_list1[i]
                        for i in range(K2-1):
                                mwp *= product_list2[-1][i]
                        for i in range(K1-1):
                                mwp *= product_list3[-1][-1][i]
                        for i in range(K1-1):
                                mwp *= primes_list[i-K1]                       
                        mw = pow(A0, mwp, n) 
                        end_time = datetime.datetime.now()     
                        print((end_time - start_time)) 

                        start_time = datetime.datetime.now() 
                        verified = pow(mw, primes_list[-1], n) == A_final                 
                        end_time = datetime.datetime.now()
                        print((end_time - start_time))
                        print(verified)


                        start_time = datetime.datetime.now() 
                        a, b = bezoute_coefficients(252533614457563255817176556954479732787, product)                             
                        d = pow(A0, a, n)
                        # return d, b          
                        end_time = datetime.datetime.now() 
                        print(d,b)    
                        print((end_time - start_time)) 

                        start_time = datetime.datetime.now() 
                        verified = (pow(d, 252533614457563255817176556954479732787, n) * pow(A_final, b, n)) % n == A0                 
                        end_time = datetime.datetime.now()
                        print((end_time - start_time))
                        print(verified)

                        print('\n')


        def test_concept_out(self):
                data = pd.read_csv('chineseconceptnet.csv', delimiter='\t')
                data.columns = ['uri', 'relation', 'start', 'end', 'json']
                weights = data['json'].apply(lambda row: json.loads(row)['weight'])
                data.pop('json')
                data.insert(4,'weights',weights)


                edgetype = set()
                ids = set()
                edges = set()
                edgesto = set()
                edgestoweight = set()
                edgesout = set()
                edgesin = set()
                C = dict()
                for i in range(len(data)):  
                        start = data.loc[i]['start']
                        end = data.loc[i]['end']
                        relation = data.loc[i]['relation']

                        edgetype.update({relation,})
                        ids.update({start, end})
                        # edges.update({start + ',' + end,})
                        # edgesto.update({start +','+ relation +','+  end,})
                        # edgestoweight.update({start +','+  relation +','+  end + ',' + str(weight),})
                        edgesout.update({start +',out,'+ relation,})
                        edgesin.update({end +',in,'+  relation,})
                        # element = start + 'to' + end + ':' + weight + ';'
                        # print(element)

                        if (start+',out,') in C:
                                C[start+',out,'+ relation].append(end);
                        else:
                                C[start+',out,'+ relation] = [end];                    
                        if (end+',in,') in C:
                                C[end+',in,'+ relation].append(start);
                        else:
                                C[end+',in,'+ relation] = [start];


                print(len(ids))
                print(len(edgetype))

                print(edgetype)

                for K in range(1,11):
                        print(K)
                        maList = []
                        nmaList = []
                        for i in range(1,11):
                                mtotal = 0
                                nmtotal = 0
                                time = 50000
                                for j in range(time):
                                        node = random.sample(ids,1)[0]
                                        # print(node)
                                        types = random.sample(edgetype,i)
                                        # print(types)
                                        m, nm = outbound(node, types, K, C)
                                        mtotal += m
                                        nmtotal += nm
                                ma = mtotal / time
                                nma = nmtotal / time
                                maList.append(ma)
                                nmaList.append(nma)
                        print(maList)
                        print(nmaList)


        def test_batch(self):    
                n = 36461482706354564422592875042006590908268153693683612285024099145347146308853
                A0 = 65537
                f = open('wiki_1m_prime.pk','rb')  
                primes_list = pickle.load(f)
                # primes_list += primes_list[:50000]
                elementTotal = primes_list[:100000]
                product = listProduct(elementTotal)
                A_final = pow(A0, product, n)  


                for total in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:
                        print(total)

                        result = primes_list[:total]

                        start_time = datetime.datetime.now()  
                        p = listProduct(result)  
                        # end_time = datetime.datetime.now() 
                        # print((end_time - start_time)) 
                        # start_time = datetime.datetime.now()  
                        mp = product // p
                        # end_time = datetime.datetime.now() 
                        # print((end_time - start_time)) 
                        # start_time = datetime.datetime.now()  
                        A_post = pow(A0, mp, n)      
                        # end_time = datetime.datetime.now() 
                        # print((end_time - start_time)) 
                        # start_time = datetime.datetime.now()                  
                        proof = prove_exponentiation_test(A_post, p, A_final, n)
                        end_time = datetime.datetime.now()   
                        print((end_time - start_time))

                        size = 0
                        size += sys.getsizeof(A_post)
                        print(size)
                        size = 0
                        size += sys.getsizeof(A_post) + sys.getsizeof(proof)
                        print(size)

                        start_time = datetime.datetime.now()
                        p = listProduct(result)  
                        verified = A_final == pow(A_post, p, n) 
                        end_time = datetime.datetime.now()     
                        print((end_time - start_time)) 
                        print(verified)

                        start_time = datetime.datetime.now()  
                        p = listProduct(result)  
                        is_valid = verify_exponentiation_test(proof, A_post, p, A_final, n)
                        end_time = datetime.datetime.now()   
                        print((end_time - start_time))
                        print(is_valid)



                        result = primes_list[-total:]
                        start_time = datetime.datetime.now()
                        p = listProduct(result)  
                        # end_time = datetime.datetime.now() 
                        # print((end_time - start_time)) 
                        # start_time = datetime.datetime.now()  
                        a, b = bezoute_coefficients(p, product)   
                        # end_time = datetime.datetime.now() 
                        # print((end_time - start_time)) 
                        # start_time = datetime.datetime.now()                           
                        d = pow(A0, a, n)
                        pre = pow(A_final, b, n) 
                        # end_time = datetime.datetime.now() 
                        # print((end_time - start_time)) 
                        # start_time = datetime.datetime.now()         
                        proof = prove_knowledge_exponent(b, A_final, pre, n)
                        end_time = datetime.datetime.now() 
                        # print(d,b)    
                        print((end_time - start_time)) 

                        size = 0
                        size += sys.getsizeof(d) + sys.getsizeof(b)
                        print(size)
                        size = 0
                        size += sys.getsizeof(d) + sys.getsizeof(pre)
                        for e in proof:
                                size += sys.getsizeof(e)
                        print(size)

                        start_time = datetime.datetime.now() 
                        p = listProduct(result)  
                        verified = (pow(d, p, n) * pow(A_final, b, n)) % n == A0                 
                        end_time = datetime.datetime.now()
                        print((end_time - start_time))
                        print(verified)

                        start_time = datetime.datetime.now()
                        p = listProduct(result)  
                        is_valid = verify_knowledge_exponent(A_final, pre, proof[0], proof[1], proof[2], n) 
                        verified = (pow(d, p, n) * pre) % n == A0             
                        end_time = datetime.datetime.now()
                        print((end_time - start_time))
                        print(is_valid)
                        print(verified)

                        print('\n')


        def test_product(self):
                f = open('wiki_1m_prime.pk','rb')  
                wiki_list = pickle.load(f)
                # primes_list += primes_list[:50000]


                for total in [1+(1<<17), 1+(1<<18), 1+(1<<19),600000, 700000, 800000, 900000, 1000000]:
                        print(total)
                        primes_list = wiki_list[:total]

 
                        start_time = datetime.datetime.now()
                        p = listProduct(primes_list)
                        end_time = datetime.datetime.now()
                        print((end_time - start_time))

                        start_time = datetime.datetime.now() 
                        q = listProductSpaced(primes_list)
                        end_time = datetime.datetime.now()
                        print((end_time - start_time))

                        print(p==q)

                        print('\n')

                        
        def test_single_witness_binary(self):               
                n = 36461482706354564422592875042006590908268153693683612285024099145347146308853
                A0 = 65537
                f = open('wiki_1m_prime.pk','rb')  
                primes_list = pickle.load(f)


                for total in [1000000, 900000, 800000, 700000, 600000, 500000, 400000, 300000, 200000, 100000]:
                        print(total)
                        primes_list = primes_list[:total]
                        p = primes_list[:]
                        if len(p)&(len(p)-1)==0:
                                t=len(p).bit_length()-1
                        else:
                                t=len(p).bit_length()
                        for i in range((1<<t)-len(p)):
                                p.append(1)

                        proResult = [p]
                        while len(p)!=2:
                                q=[]
                                for i in range(len(p)//2):
                                        q.append(p[2*i]*p[2*i+1])
                                proResult.append(q)
                                p=q
                        product = p[0]*p[1] 
                        A_final = pow(A0, product, n) 


                        mwp = 1
                        sample = 333
                        k=3
                        p = primes_list[:]
                        p[sample] = 1
                        start_time = datetime.datetime.now() 
                        if len(p)&(len(p)-1)==0:
                                t=len(p).bit_length()-1
                        else:
                                t=len(p).bit_length()
                        for i in range((1<<t)-len(p)):
                                p.append(1)
                        s = sample>>(t-k)
                        mwp = listProduct(p[s<<(t-k):(s+1)<<(t-k)])
                        for r in range(k):
                                if (s>>r)%2==1:
                                        mwp *= proResult[-k+r][((s>>(r+1))<<1)]
                                else:
                                        mwp *= proResult[-k+r][((s>>(r+1))<<1)+1]                   
                        mw = pow(A0, mwp, n) 
                        end_time = datetime.datetime.now()     
                        print((end_time - start_time)) 

                        start_time = datetime.datetime.now() 
                        verified = pow(mw, primes_list[sample], n) == A_final                 
                        end_time = datetime.datetime.now()
                        print((end_time - start_time))
                        print(verified)


                        start_time = datetime.datetime.now() 
                        a, b = bezoute_coefficients(252533614457563255817176556954479732787, product)                             
                        d = pow(A0, a, n)
                        # return d, b          
                        end_time = datetime.datetime.now() 
                        print(d,b)    
                        print((end_time - start_time)) 

                        start_time = datetime.datetime.now() 
                        verified = (pow(d, 252533614457563255817176556954479732787, n) * pow(A_final, b, n)) % n == A0                 
                        end_time = datetime.datetime.now()
                        print((end_time - start_time))
                        print(verified)

                        print('\n')

               
        def test_single_witness_division(self):               
                n = 36461482706354564422592875042006590908268153693683612285024099145347146308853
                A0 = 65537
                f = open('wiki_1m_prime.pk','rb')  
                primes_list = pickle.load(f)


                for total in [100000]:
                        print(total)
                        primes_list = primes_list[:total]

                        start_time = datetime.datetime.now() 
                        product = listProduct(primes_list)
                        end_time = datetime.datetime.now()     
                        print((end_time - start_time)) 

                        A_final = pow(A0, product, n) 


                        sample = 333    
                        start_time = datetime.datetime.now() 
                        mwp = product//primes_list[sample]         
                        mw = pow(A0, mwp, n) 
                        end_time = datetime.datetime.now()     
                        print((end_time - start_time)) 

                        start_time = datetime.datetime.now() 
                        verified = False
                        # for i in range(10000):
                        verified = pow(mw, primes_list[sample], n) == A_final                 
                        end_time = datetime.datetime.now()
                        print((end_time - start_time))
                        print(verified)


                        start_time = datetime.datetime.now() 
                        a, b = bezoute_coefficients(252533614457563255817176556954479732787, product)                             
                        d = pow(A0, a, n)
                        # return d, b          
                        end_time = datetime.datetime.now() 
                        # print(d,b)    
                        print((end_time - start_time)) 

                        start_time = datetime.datetime.now() 
                        verified = False
                        # for i in range(10000):
                        verified = (pow(d, 252533614457563255817176556954479732787, n) * pow(A_final, b, n)) % n == A0                 
                        end_time = datetime.datetime.now()
                        print((end_time - start_time))
                        print(verified)

                        print('\n')

               
        def test_batch_division(self):               
                n = 36461482706354564422592875042006590908268153693683612285024099145347146308853
                A0 = 65537
                f = open('wiki_1m_prime.pk','rb')  
                primes_list = pickle.load(f)
                # primes_list += primes_list[:50000]
                elementTotal = primes_list[:100000]
                product = listProduct(elementTotal)
                A_final = pow(A0, product, n)  


                for total in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:
                        print(total)
                        result = primes_list[:total]                       
                        p = listProduct(result)

                        # start_time = datetime.datetime.now()  
                        # mw_list = []
                        # for e in result:                  
                        #         mp = product // e 
                        #         mw_list.append(pow(A0, mp, n))                     
                        # end_time = datetime.datetime.now()   
                        # print((end_time - start_time))
                        # size = 0
                        # for e in mw_list:
                        #         size += sys.getsizeof(e)
                        # print(size)

                        start_time = datetime.datetime.now()  
                        p = listProduct(result)  
                        mp = product // p 
                        A_post = pow(A0, mp, n)                       
                        end_time = datetime.datetime.now()   
                        print((end_time - start_time))
                        size = 0
                        size += sys.getsizeof(A_post)
                        print(size)

                        start_time = datetime.datetime.now()  
                        p = listProduct(result)  
                        mp = product // p 
                        A_post = pow(A0, mp, n)                       
                        Q = prove_exponentiation_test(A_post, p, A_final, n)
                        end_time = datetime.datetime.now()   
                        print((end_time - start_time))
                        size = 0
                        size += sys.getsizeof(A_post) + sys.getsizeof(Q)
                        print(size)


                        # start_time = datetime.datetime.now()  
                        # verified = True
                        # for i in range(len(mw_list)): 
                        #         if verified:        
                        #                 verified = A_final == pow(mw_list[i], result[i], n)                    
                        # end_time = datetime.datetime.now()   
                        # print((end_time - start_time))
                        # print(verified)

                        start_time = datetime.datetime.now()
                        p = listProduct(result)  
                        verified = A_final == pow(A_post, p, n) 
                        end_time = datetime.datetime.now()     
                        print((end_time - start_time)) 
                        print(verified)

                        start_time = datetime.datetime.now()  
                        p = listProduct(result)  
                        is_valid = verify_exponentiation_test(Q, A_post, p, A_final, n)
                        end_time = datetime.datetime.now()   
                        print((end_time - start_time))
                        print(is_valid)
                        print('\n')



                        result = primes_list[-total:]
                        p = listProduct(result) 
                        start_time = datetime.datetime.now()
                        p = listProduct(result)  
                        a, b = bezoute_coefficients(p, product)                              
                        d = pow(A0, a, n)  
                        end_time = datetime.datetime.now() 
                        # print(d,b)    
                        print((end_time - start_time)) 
                        size = 0
                        size += sys.getsizeof(d) + sys.getsizeof(b)
                        print(size)

                        result = primes_list[-total:]
                        start_time = datetime.datetime.now()
                        p = listProduct(result)  
                        a, b = bezoute_coefficients(p, product)                              
                        d = pow(A0, a, n)
                        pre = pow(A_final, b, n) 
                        pre_inv = mul_inv(pre, n) 
                        # print(pre_inv)      
                        proof = prove_knowledge_exponent(b, A_final, pre, n)
                        Q = prove_exponentiation_test(d, p, (A0*pre_inv)%n, n)
                        # print(pow(d,p,n)==(A0*pre_inv)%n) 
                        end_time = datetime.datetime.now() 
                        # print(d,b)    
                        print((end_time - start_time)) 
                        size = 0
                        size += sys.getsizeof(d) + sys.getsizeof(pre)
                        for e in proof:
                                size += sys.getsizeof(e)
                        size += sys.getsizeof(Q)
                        print(size)

                        start_time = datetime.datetime.now() 
                        p = listProduct(result)  
                        verified = (pow(d, p, n) * pow(A_final, b, n)) % n == A0                 
                        end_time = datetime.datetime.now()
                        print((end_time - start_time))
                        print(verified)

                        start_time = datetime.datetime.now()
                        p = listProduct(result)  
                        is_valid = verify_knowledge_exponent(A_final, pre, proof[0], proof[1], proof[2], n) 
                        pre_inv = mul_inv(pre, n)       
                        verified = verify_exponentiation_test(Q, d, p, (A0*pre_inv)%n, n)
                        # verified = (pow(d, p, n) * pre) % n == A0             
                        end_time = datetime.datetime.now()
                        print((end_time - start_time))
                        print(is_valid)
                        print(verified)

                        print('\n')
   

        def test_deletion(self):                            
                p = 252533614457563255817176556954479732787
                q = 144382690536755041477176940054403505719
                n = 36461482706354564422592875042006590908268153693683612285024099145347146308853
                # draw random number within range of [0,n-1]
                A0 = 65537
                f = open('wiki_1m_prime.pk','rb')  
                primes_list = pickle.load(f)
                samples = primes_list[:10]

 
                product = listProduct(samples) 
                A_final = pow(A0, product, n) 

                sample = 5  
                # print(primes_list[sample])
                sam_inv = mul_inv(primes_list[sample], (p-1)*(q-1))
                # print(sam_inv)
                A_del = pow(A_final, sam_inv, n) 
                # print(pow(A_del, primes_list[sample], n) == A_final)
  
                mwp = product//primes_list[sample]         
                mw = pow(A0, mwp, n) 
                print(A_del==mw)



def outbound(id, tList, K, C): 
        m = 0
        nm = 0      
        for t in tList:
                if (id+',out,'+t) in C:
                        m += 1
                        nextList = C[id+',out,'+t]
                        if K>1:
                                for node in nextList:
                                        mr, nmr = outbound(node, tList, K-1, C)
                                        m += mr
                                        nm += nmr
                else:
                        nm += 1
        return m, nm

def listProduct(p):
        if len(p)&(len(p)-1)==0:
                t=len(p).bit_length()-1
        else:
                t=len(p).bit_length()

        for i in range((1<<t)-len(p)):
                p.append(1)
        while len(p)!=1:
                q=[]
                for i in range(len(p)//2):
                        q.append(p[2*i]*p[2*i+1])
                # size = 0
                # for e in q:
                #         size += sys.getsizeof(e)
                # print(size)
                p=q
        return p[0]

def listProductSpaced(p):
        if len(p)&(len(p)-1)==0:
                t=len(p).bit_length()-1
        else:
                t=len(p).bit_length()

        for i in range((1<<t)-len(p)):
                p.insert(2*i,1)
        while len(p)!=1:
                q=[]
                for i in range(len(p)//2):
                        q.append(p[2*i]*p[2*i+1])
                p=q
        return p[0]



#print(hash_to_prime(10))
s = unittest.TestSuite()
testname = 'test'
print(testname)
s.addTest(AccumulatorTest(testname))
#s.addTests([Test_Myclass1("test_sub"),Test_Myclass1("test_sum")])
fs = open(testname+'.txt',"w")
run = unittest.TextTestRunner(fs)
run.run(s)