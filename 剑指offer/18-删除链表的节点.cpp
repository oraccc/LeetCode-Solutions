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
    ListNode* deleteNode(ListNode* head, int val) {
        ListNode* dummyHead = new ListNode(-1);
        dummyHead->next = head;
        ListNode* prev = dummyHead;
        while (prev->next) {
            if (prev->next->val == val) {
                prev->next = prev->next->next;
                break;
            }
            prev = prev->next;
        }
        return dummyHead->next;
    }
};