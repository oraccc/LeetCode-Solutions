/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        ListNode* dummyHead = new ListNode(-1);
        ListNode* curr = dummyHead;
        while (l1 && l2) {
            if (l1->val < l2->val) {
                curr->next = new ListNode(l1->val);
                l1 = l1->next;
            }
            else {
                curr->next = new ListNode(l2->val);
                l2 = l2->next;
            }
            curr = curr->next;
        }
        if (l1) {
            curr->next = l1;
        }
        else if (l2) {
            curr->next = l2;
        }
        return dummyHead->next;
    }
};